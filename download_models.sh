#!/bin/bash
# Download all model checkpoints needed for MLSys benchmarking.
# Run this from the ExecuTorch repo root.
#
# Models that auto-download (torchvision pretrained weights):
#   - MobileNetV3, ViT, EfficientNet, Swin — downloaded during export, no action needed.
#
# Models that need manual download:
#   - Llama 3.2 1B/3B — Meta checkpoints (llama CLI or llama.com)
#   - Qwen 3 0.6B — HuggingFace (auto-downloaded during export, but pre-caching here)
#   - MobileBERT, DistilBERT — HuggingFace transformers models

set -euo pipefail

LLAMA_CHECKPOINT_DIR="$HOME/.llama/checkpoints"

echo "========================================="
echo " 1. Llama 3.2 1B Instruct"
echo "========================================="
if [ -f "$LLAMA_CHECKPOINT_DIR/Llama3.2-1B-Instruct/consolidated.00.pth" ]; then
    echo "✔ Already downloaded at $LLAMA_CHECKPOINT_DIR/Llama3.2-1B-Instruct/"
else
    echo "✘ Not found. Download from https://www.llama.com/llama-downloads/"
    echo "  Or run: llama download --source meta --model-id Llama3.2-1B-Instruct"
    echo ""
fi

echo ""
echo "========================================="
echo " 2. Llama 3.2 3B Instruct"
echo "========================================="
if [ -f "$LLAMA_CHECKPOINT_DIR/Llama3.2-3B-Instruct/consolidated.00.pth" ]; then
    echo "✔ Already downloaded at $LLAMA_CHECKPOINT_DIR/Llama3.2-3B-Instruct/"
else
    echo "✘ Not found. Download from https://www.llama.com/llama-downloads/"
    echo "  Or run: llama download --source meta --model-id Llama3.2-3B-Instruct"
    echo ""
fi

echo ""
echo "========================================="
echo " 3. Qwen 3 0.6B (HuggingFace)"
echo "========================================="
echo "Pre-downloading Qwen 3 0.6B from HuggingFace..."
echo "(This will be auto-converted to Meta format during export)"
hf download Qwen/Qwen3-0.6B

echo ""
echo "========================================="
echo " 4. MobileBERT (HuggingFace)"
echo "========================================="
echo "Pre-downloading MobileBERT from HuggingFace..."
hf download google/mobilebert-uncased

echo ""
echo "========================================="
echo " 5. DistilBERT (HuggingFace)"
echo "========================================="
echo "Pre-downloading DistilBERT from HuggingFace..."
hf download distilbert/distilbert-base-uncased

echo ""
echo "========================================="
echo " 6. Vision models (auto-download)"
echo "========================================="
echo "The following models use torchvision pretrained weights"
echo "and will be downloaded automatically during export:"
echo "  - MobileNetV3"
echo "  - ViT"
echo "  - EfficientNet-B0"
echo "  - Swin Transformer Tiny"
echo ""
echo "No manual download needed."

echo ""
echo "========================================="
echo " Summary"
echo "========================================="
echo "Llama 3.2 1B:  $([ -f "$LLAMA_CHECKPOINT_DIR/Llama3.2-1B-Instruct/consolidated.00.pth" ] && echo '✔ Ready' || echo '✘ Missing')"
echo "Llama 3.2 3B:  $([ -f "$LLAMA_CHECKPOINT_DIR/Llama3.2-3B-Instruct/consolidated.00.pth" ] && echo '✔ Ready' || echo '✘ Missing')"
echo "Qwen 3 0.6B:   ✔ Downloaded (will convert during export)"
echo "MobileBERT:    ✔ Downloaded"
echo "DistilBERT:    ✔ Downloaded"
echo "MobileNetV3:   ✔ Auto-download during export"
echo "ViT:           ✔ Auto-download during export"
echo "EfficientNet:  ✔ Auto-download during export"
echo "Swin:          ✔ Auto-download during export"
