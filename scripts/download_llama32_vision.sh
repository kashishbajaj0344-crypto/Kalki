#!/bin/bash
#!/bin/bash
# Download Llama 3.2 Vision 11B from Hugging Face
# Converts to MLX format with int4 quantization for optimal performance on M4 Max
# Requires: HF access approved (you have it!)

set -e  # Exit on error

echo "🚀 Llama 3.2 Vision 11B Download & Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running from Kalki directory
if [ ! -f "kalki_cli.py" ]; then
    echo -e "${RED}Error: Must run from Kalki root directory${NC}"
    exit 1
fi

# Create models directory
mkdir -p models

# Step 1: Check/Install dependencies
echo -e "${YELLOW}Step 1/4: Checking dependencies...${NC}"
pip install -q huggingface_hub mlx mlx-lm || {
    echo -e "${RED}Failed to install dependencies${NC}"
    exit 1
}

# Step 2: Check HF login
echo -e "${YELLOW}Step 2/4: Verifying Hugging Face authentication...${NC}"
if ! huggingface-cli whoami &>/dev/null; then
    echo -e "${RED}Not logged into Hugging Face!${NC}"
    echo "Run: huggingface-cli login"
    echo "Paste your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

echo -e "${GREEN}✓ Logged in as: $(huggingface-cli whoami | head -1)${NC}"

# Step 3: Download from Hugging Face
echo -e "${YELLOW}Step 3/4: Downloading Llama 3.2 Vision 11B (~22 GB)...${NC}"
echo "This will take 10-30 minutes depending on your connection."
echo ""

python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import os

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
local_dir = "models/llama-3.2-11b-vision-hf"

print(f"Downloading {model_id}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)
print(f"✓ Downloaded to: {local_dir}")
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Download failed. Check your HF access and internet.${NC}"
    exit 1
fi

# Step 4: Convert to MLX + Quantize to int4
echo -e "${YELLOW}Step 4/4: Converting to MLX format + int4 quantization...${NC}"
echo "This optimizes for M4 Max and reduces RAM usage by 75%"
echo ""

python3 -m mlx_lm.convert \
    --hf-path models/llama-3.2-11b-vision-hf \
    --mlx-path models/llama-3.2-11b-vision-int4 \
    --quantize \
    -q 4

if [ $? -ne 0 ]; then
    echo -e "${RED}Conversion failed. You may need to update mlx-lm:${NC}"
    echo "pip install -U mlx-lm"
    exit 1
fi

# Cleanup: Remove HF format to save space (optional)
echo ""
read -p "Delete original HF format to save ~16GB? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf models/llama-3.2-11b-vision-hf
    echo -e "${GREEN}✓ Cleaned up HF format${NC}"
fi

# Final summary
echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Model location: models/llama-3.2-11b-vision-int4"
echo "Format: MLX int4 (optimized for Apple Silicon)"
echo "Memory usage: ~6-8 GB (vs 22 GB original)"
echo ""
echo "Quick test:"
echo "  python3 -c 'from mlx_lm import load; load(\"models/llama-3.2-11b-vision-int4\")'"
echo ""
echo "Next: Update kalki to use vision model for PDF ingestion"
echo "=========================================="
# For use on Apple Silicon M4 Max

set -e  # Exit on error

echo "=================================="
echo "🦙 Llama 3.2 Vision 11B Download"
echo "=================================="
echo ""

# Check if llama-stack is installed
if ! command -v llama &> /dev/null; then
    echo "📦 Installing llama-stack..."
    pip install llama-stack -U
    echo "✅ llama-stack installed"
else
    echo "✅ llama-stack already installed"
fi

echo ""
echo "📋 Available models:"
llama model list | head -20

echo ""
echo "=================================="
echo "Starting download..."
echo "=================================="
echo ""
echo "⚠️  When prompted, paste your custom URL:"
echo ""
echo "https://llama3-2-multimodal.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiYTZxbHh0aTVrMjNlMHI5MDVyMDNoZGh3IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbXVsdGltb2RhbC5sbGFtYW1ldGEubmV0XC8qIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzYyOTE2NjYwfX19XX0_&Signature=CJfYqLYXbAbKot7G4huD0w8U8vxCHvZHTwtodc90QFA92wJtOwWCN8n2npJ8Cn1t8IT1vYEYr6nuoHjKt1sS4rAS%7EmQN0jvaJOJ3aEc4Id3GBa8kIKeGiAB1EPlrrIEbchhp5Ufnid9Jn7woO5BgzQnWeSe6Lx0b7fOFir6C6Hq-JHeYARzasLF0UiHUKUUFewafI25e12JgkMFuLa99E6JBNoBnz1PpSAEEt16MZBr0J2MzuxAYFADHg1iqHEMZg1UuLT0b4S15q9CKSaFfh3mbUk754BqXSbPOq-dGm78JJY%7EIXSJqTmateFjw95x8J%7EokU4ocZQpOICSSv0vMnw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1622494765825056"
echo ""
echo "⏰ URL expires in 48 hours (5 downloads max)"
echo ""

# Run the download
llama model download --source meta --model-id Llama3.2-11B-Vision-Instruct

echo ""
echo "=================================="
echo "✅ Download Complete!"
echo "=================================="
echo ""
echo "📁 Model location:"
echo "   ~/.llama/checkpoints/Llama3.2-11B-Vision-Instruct"
echo ""
echo "💾 Size: ~22 GB"
echo ""
echo "🔧 Next steps:"
echo ""
echo "1. Convert to MLX format + quantize (saves 70% RAM):"
echo ""
echo "   pip install mlx-lm"
echo "   python -m mlx_lm.convert \\"
echo "     --hf-path ~/.llama/checkpoints/Llama3.2-11B-Vision-Instruct \\"
echo "     --mlx-path models/llama-3.2-11b-vision-int4 \\"
echo "     --quantize -q 4"
echo ""
echo "2. Test the model:"
echo ""
echo "   python models_config.py"
echo ""
echo "=================================="
