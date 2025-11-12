#!/usr/bin/env python3
"""
Fine-tune Llama 3.1 8B on Construction Knowledge using MLX LoRA
Optimized for Apple Silicon M4 Max

This is a WORKING wrapper around MLX's official LoRA fine-tuning CLI.

Usage:
    # Basic (uses defaults)
    python finetune_llama_working.py --training-data data/training/training_data_20251108.jsonl
    
    # Advanced (custom hyperparameters)
    python finetune_llama_working.py \
        --training-data data/training/training_data_20251108.jsonl \
        --model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct \
        --iters 1000 \
        --batch-size 4 \
        --lora-layers 16 \
        --learning-rate 1e-5
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict


def validate_training_data(file_path: str) -> tuple[bool, int]:
    """
    Validate JSONL training data format
    
    Returns:
        (is_valid, num_examples)
    """
    try:
        with open(file_path, 'r') as f:
            examples = [json.loads(line) for line in f if line.strip()]
        
        # Check format
        required_keys = {'instruction', 'input', 'output'}
        for i, ex in enumerate(examples[:5]):  # Check first 5
            if not required_keys.issubset(ex.keys()):
                print(f"❌ Example {i} missing required keys: {required_keys - ex.keys()}")
                return False, 0
        
        return True, len(examples)
    except Exception as e:
        print(f"❌ Error reading training data: {e}")
        return False, 0


def convert_to_mlx_format(input_file: str, output_file: str) -> bool:
    """
    Convert Kalki JSONL format to MLX LoRA expected format
    
    MLX expects: {"text": "full_conversation"}
    Kalki has: {"instruction": "...", "input": "...", "output": "..."}
    """
    print(f"📝 Converting {input_file} to MLX format...")
    
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                
                example = json.loads(line)
                
                # Format as Llama 3.1 Instruct conversation
                instruction = example.get('instruction', '')
                input_text = example.get('input', '')
                output = example.get('output', '')
                
                if input_text:
                    full_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a construction engineering expert assistant specializing in building codes, structural design, and construction best practices.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
                else:
                    full_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a construction engineering expert assistant specializing in building codes, structural design, and construction best practices.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
                
                mlx_example = {"text": full_text}
                f_out.write(json.dumps(mlx_example) + '\n')
        
        print(f"✅ Converted to {output_file}")
        return True
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False


def run_mlx_lora_training(
    model_path: str,
    train_file: str,
    output_dir: str,
    iters: int = 1000,
    batch_size: int = 4,
    lora_layers: int = 16,
    learning_rate: float = 1e-5,
    val_batches: int = 25,
    save_every: int = 100,
) -> bool:
    """
    Run MLX LoRA fine-tuning using the official CLI
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_path,
        "--train",
        "--data", train_file,
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--lora-layers", str(lora_layers),
        "--learning-rate", str(learning_rate),
        "--val-batches", str(val_batches),
        "--save-every", str(save_every),
        "--adapter-path", str(output_path / "adapters"),
    ]
    
    print("\n" + "="*70)
    print("🚀 Starting MLX LoRA Fine-Tuning")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print("="*70)
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print("\n❌ mlx_lm.lora module not found!")
        print("Install with: pip install mlx-lm")
        return False


def test_finetuned_model(model_path: str, adapter_path: str):
    """
    Test the fine-tuned model with a sample prompt
    """
    print("\n" + "="*70)
    print("🧪 Testing Fine-Tuned Model")
    print("="*70)
    
    try:
        from mlx_lm import load, generate
        
        print(f"Loading model from: {model_path}")
        print(f"Loading adapters from: {adapter_path}")
        
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        
        test_prompt = """What is the minimum footing depth required by IRC for residential foundations in frost-prone regions?"""
        
        print(f"\n📝 Test Prompt:\n{test_prompt}\n")
        print("💭 Generating response...\n")
        
        response = generate(
            model, 
            tokenizer, 
            prompt=test_prompt,
            max_tokens=200,
            temp=0.7,
        )
        
        print(f"🤖 Response:\n{response}\n")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.1 8B on construction knowledge (MLX wrapper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--training-data',
        type=str,
        required=True,
        help='Path to Kalki training data JSONL file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct',
        help='Path to base Llama 3.1 8B model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models/llama_finetuned',
        help='Output directory for LoRA adapters'
    )
    parser.add_argument(
        '--iters',
        type=int,
        default=1000,
        help='Number of training iterations'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size (4-8 recommended for 8B on M4 Max)'
    )
    parser.add_argument(
        '--lora-layers',
        type=int,
        default=16,
        help='Number of layers to apply LoRA (16 = all for 8B)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the fine-tuned model after training'
    )
    
    args = parser.parse_args()
    
    # Expand home directory
    model_path = Path(args.model).expanduser()
    training_data = Path(args.training_data).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    
    print("="*70)
    print("🔥 Llama 3.1 8B LoRA Fine-Tuning (Apple Silicon M4 Max)")
    print("="*70)
    print(f"📦 Base model: {model_path}")
    print(f"📚 Training data: {training_data}")
    print(f"📁 Output: {output_dir}")
    print(f"⚙️  Iterations: {args.iters}")
    print(f"⚙️  Batch size: {args.batch_size}")
    print(f"⚙️  LoRA layers: {args.lora_layers}")
    print(f"⚙️  Learning rate: {args.learning_rate}")
    print("="*70)
    print()
    
    # Validate training data exists
    if not training_data.exists():
        print(f"❌ Training file not found: {training_data}")
        print("\n💡 Generate training data first:")
        print("   python kalki_cli.py learn training")
        sys.exit(1)
    
    # Validate model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("\n💡 Possible locations:")
        print("   ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct")
        print("   ~/.llama/checkpoints/Llama3.1-8B-Instruct")
        sys.exit(1)
    
    # Validate training data format
    print("🔍 Validating training data...")
    is_valid, num_examples = validate_training_data(str(training_data))
    
    if not is_valid:
        print("❌ Training data validation failed")
        sys.exit(1)
    
    print(f"✅ Found {num_examples} valid training examples")
    
    # Convert to MLX format
    mlx_train_file = output_dir / "train_mlx_format.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not convert_to_mlx_format(str(training_data), str(mlx_train_file)):
        print("❌ Failed to convert training data")
        sys.exit(1)
    
    # Run training
    success = run_mlx_lora_training(
        model_path=str(model_path),
        train_file=str(mlx_train_file),
        output_dir=str(output_dir),
        iters=args.iters,
        batch_size=args.batch_size,
        lora_layers=args.lora_layers,
        learning_rate=args.learning_rate,
    )
    
    if not success:
        print("\n❌ Training failed")
        sys.exit(1)
    
    print("\n✅ Fine-tuning complete!")
    print(f"📁 LoRA adapters saved to: {output_dir / 'adapters'}")
    
    # Test if requested
    if args.test:
        adapter_path = output_dir / "adapters"
        if adapter_path.exists():
            test_finetuned_model(str(model_path), str(adapter_path))
        else:
            print(f"⚠️  Adapter path not found: {adapter_path}")
    
    print("\n" + "="*70)
    print("📖 Usage Instructions")
    print("="*70)
    print("\nLoad fine-tuned model in Python:")
    print(f"""
from mlx_lm import load, generate

model, tokenizer = load(
    "{model_path}",
    adapter_path="{output_dir / 'adapters'}"
)

response = generate(model, tokenizer, "Your prompt here", max_tokens=200)
print(response)
    """)
    print("="*70)


if __name__ == "__main__":
    main()
