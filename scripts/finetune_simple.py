#!/usr/bin/env python3
"""
Simple Fine-tune Script for Llama 3.1 8B using MLX LoRA
Optimized for Apple Silicon M4 Max

This uses MLX's built-in LoRA fine-tuning which is production-ready.

Usage:
    # Step 1: Generate training data from PDFs
    python kalki_cli.py learn training
    
    # Step 2: Fine-tune with MLX LoRA
    python finetune_simple.py

Requirements:
    pip install mlx mlx-lm
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def convert_to_mlx_format(input_file: str, output_file: str):
    """
    Convert Kalki training data to MLX LoRA format
    
    MLX expects format:
    {"text": "full formatted prompt + response"}
    
    Kalki format:
    {"instruction": "...", "input": "...", "output": "..."}
    """
    print(f"📝 Converting training data to MLX format...")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")
    
    converted_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            item = json.loads(line.strip())
            
            # Format as Llama 3.1 Instruct
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output = item.get('output', '')
            
            if input_text:
                formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a construction engineering expert assistant specializing in building codes, structural design, and construction best practices.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
            else:
                formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a construction engineering expert assistant specializing in building codes, structural design, and construction best practices.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
            
            # Write in MLX format
            f_out.write(json.dumps({"text": formatted}) + '\n')
            converted_count += 1
    
    print(f"✅ Converted {converted_count} examples")
    return converted_count


def run_mlx_lora_finetuning(
    model: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    train_file: str = "data/training/train_mlx.jsonl",
    val_file: str = "data/training/valid_mlx.jsonl",
    adapter_path: str = "data/models/lora_adapters",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    lora_rank: int = 8,
):
    """
    Run MLX LoRA fine-tuning using mlx_lm.lora
    
    This uses Apple's official MLX LoRA implementation which is:
    - Highly optimized for M-series chips
    - Memory efficient
    - Production ready
    """
    print("\n" + "="*60)
    print("🔥 Starting MLX LoRA Fine-Tuning")
    print("="*60)
    print(f"📦 Base model: {model}")
    print(f"📚 Training data: {train_file}")
    print(f"📁 Output: {adapter_path}")
    print(f"⚙️  Epochs: {num_epochs}")
    print(f"⚙️  Batch size: {batch_size}")
    print(f"⚙️  Learning rate: {learning_rate}")
    print(f"⚙️  LoRA rank: {lora_rank}")
    print("="*60 + "\n")
    
    # Build command
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", model,
        "--train",
        "--data", train_file,
        "--adapter-path", adapter_path,
        "--iters", str(num_epochs * 1000),  # Approximate iterations
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--lora-layers", "16",  # Apply LoRA to 16 layers
        "--rank", str(lora_rank),
    ]
    
    if Path(val_file).exists():
        cmd.extend(["--val-data", val_file])
    
    print(f"🚀 Running command:")
    print(f"   {' '.join(cmd)}\n")
    
    # Run fine-tuning
    result = subprocess.run(cmd, check=False)
    
    if result.returncode == 0:
        print("\n✅ Fine-tuning completed successfully!")
        print(f"📁 LoRA adapters saved to: {adapter_path}")
        return True
    else:
        print("\n❌ Fine-tuning failed!")
        return False


def test_finetuned_model(model: str, adapter_path: str):
    """Test the fine-tuned model with construction questions"""
    print("\n" + "="*60)
    print("🧪 Testing Fine-Tuned Model")
    print("="*60)
    
    test_prompts = [
        "What is the maximum span for a 2x8 floor joist at 16 inches on-center?",
        "What are the requirements for foundation depth in cold climates?",
        "Explain the formula for calculating beam deflection.",
    ]
    
    # Use mlx_lm to generate with LoRA adapter
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}\n")
        
        # Build generation command
        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", model,
            "--adapter-path", adapter_path,
            "--prompt", prompt,
            "--max-tokens", "200",
            "--temp", "0.7",
        ]
        
        print("Generating response...")
        subprocess.run(cmd)
        print()


def main():
    print("="*60)
    print("🏗️  Kalki Construction AI - Fine-Tuning Pipeline")
    print("="*60)
    
    # Step 1: Check if training data exists
    training_dir = Path("data/training")
    training_files = list(training_dir.glob("training_data_*.jsonl"))
    
    if not training_files:
        print("\n❌ No training data found!")
        print("\nGenerate training data first:")
        print("  python kalki_cli.py learn training")
        return
    
    latest_training = max(training_files, key=lambda p: p.stat().st_mtime)
    print(f"\n✅ Found training data: {latest_training}")
    
    # Count examples
    with open(latest_training) as f:
        num_examples = sum(1 for _ in f)
    print(f"📊 Total examples: {num_examples}")
    
    if num_examples < 10:
        print("\n⚠️  Warning: Very few training examples!")
        print("   Consider ingesting more PDFs first.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 2: Convert to MLX format
    mlx_train_file = training_dir / "train_mlx.jsonl"
    mlx_val_file = training_dir / "valid_mlx.jsonl"
    
    print("\n" + "="*60)
    print("📝 Step 1: Convert to MLX Format")
    print("="*60)
    
    # Split 90/10 train/val
    all_data = []
    with open(latest_training) as f:
        all_data = [line for line in f]
    
    val_size = max(1, int(len(all_data) * 0.1))
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]
    
    print(f"📊 Split: {len(train_data)} train / {len(val_data)} validation")
    
    # Write splits
    temp_train = training_dir / "temp_train.jsonl"
    temp_val = training_dir / "temp_val.jsonl"
    
    with open(temp_train, 'w') as f:
        f.writelines(train_data)
    with open(temp_val, 'w') as f:
        f.writelines(val_data)
    
    # Convert both
    convert_to_mlx_format(str(temp_train), str(mlx_train_file))
    convert_to_mlx_format(str(temp_val), str(mlx_val_file))
    
    # Clean up temps
    temp_train.unlink()
    temp_val.unlink()
    
    # Step 3: Run MLX LoRA fine-tuning
    print("\n" + "="*60)
    print("🔥 Step 2: Fine-Tune with LoRA")
    print("="*60)
    
    adapter_path = "data/models/lora_adapters"
    
    success = run_mlx_lora_finetuning(
        model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        train_file=str(mlx_train_file),
        val_file=str(mlx_val_file),
        adapter_path=adapter_path,
        num_epochs=3,
        batch_size=4,
        learning_rate=1e-4,
        lora_rank=8,
    )
    
    if not success:
        print("\n💡 Tip: MLX LoRA might not be installed correctly.")
        print("   Try: pip install --upgrade mlx mlx-lm")
        return
    
    # Step 4: Test the model
    print("\n" + "="*60)
    print("🧪 Step 3: Test Fine-Tuned Model")
    print("="*60)
    
    response = input("\nTest the fine-tuned model? (y/n): ")
    if response.lower() == 'y':
        test_finetuned_model(
            model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            adapter_path=adapter_path
        )
    
    # Step 5: Usage instructions
    print("\n" + "="*60)
    print("✅ Fine-Tuning Complete!")
    print("="*60)
    print(f"\n📁 LoRA adapters saved to: {adapter_path}")
    print("\n🚀 To use the fine-tuned model:")
    print(f"\n1. Via Python:")
    print(f"""
from mlx_lm import load, generate

model, tokenizer = load(
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    adapter_path="{adapter_path}"
)

response = generate(
    model, tokenizer,
    prompt="Your construction question here",
    max_tokens=200
)
print(response)
""")
    
    print(f"\n2. Via Command Line:")
    print(f"""
python -m mlx_lm.generate \\
    --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
    --adapter-path {adapter_path} \\
    --prompt "Your construction question here"
""")
    
    print(f"\n3. Update Kalki to use fine-tuned model:")
    print(f"""
Edit modules/llm.py and add adapter_path to load():
    model, tokenizer = load(model_id, adapter_path="{adapter_path}")
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Fine-tuning interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
