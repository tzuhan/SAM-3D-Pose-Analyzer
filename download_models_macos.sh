#!/bin/bash
# download_models_macos.sh - macOS model download script

echo "📦 SAM 3D Pose Analyzer - Downloading models for macOS..."

# Load HF_TOKEN from .env file if it exists
if [ -f ".env" ]; then
    echo "📄 Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# 1. Clone external repositories
echo ""
echo "=========================================="
echo "Step 1: Cloning external repositories..."
echo "=========================================="
mkdir -p repos
cd repos

if [ ! -d "sam-3d-body" ]; then
    echo "Cloning sam-3d-body..."
    git clone https://github.com/facebookresearch/sam-3d-body.git
else
    echo "✓ sam-3d-body already exists"
fi

if [ ! -d "sam3" ]; then
    echo "Cloning sam3..."
    git clone https://github.com/facebookresearch/sam3.git
else
    echo "✓ sam3 already exists"
fi

if [ ! -d "MoGe" ]; then
    echo "Cloning MoGe..."
    git clone https://github.com/microsoft/MoGe.git
else
    echo "✓ MoGe already exists"
fi

cd ..

# 2. Download models from HuggingFace
echo ""
echo "=========================================="
echo "Step 2: Downloading model weights..."
echo "=========================================="
mkdir -p weights/assets

python3 << 'EOF'
import os
import shutil
from huggingface_hub import hf_hub_download, login

def download():
    # Login with HF_TOKEN if available
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("🔑 Using HF_TOKEN from environment...")
        login(token=hf_token)
    else:
        print("⚠️  No HF_TOKEN found. Using cached credentials or anonymous access.")

    # Create directories
    os.makedirs("weights/assets", exist_ok=True)

    # SAM 3D Body models - download and copy to correct location
    print("Downloading SAM 3D Body models...")

    # Download model.ckpt
    ckpt_path = hf_hub_download(
        repo_id="facebook/sam-3d-body-dinov3",
        filename="model.ckpt"
    )
    shutil.copy(ckpt_path, "weights/model.ckpt")
    print(f"✓ Copied model.ckpt to weights/")

    # Download mhr_model.pt
    mhr_path = hf_hub_download(
        repo_id="facebook/sam-3d-body-dinov3",
        filename="assets/mhr_model.pt"
    )
    shutil.copy(mhr_path, "weights/assets/mhr_model.pt")
    print(f"✓ Copied mhr_model.pt to weights/assets/")

    # Download model_config.yaml
    cfg_path = hf_hub_download(
        repo_id="facebook/sam-3d-body-dinov3",
        filename="model_config.yaml"
    )
    shutil.copy(cfg_path, "weights/model_config.yaml")
    print(f"✓ Copied model_config.yaml to weights/")

    print("✅ Model download complete!")

if __name__ == "__main__":
    download()
EOF

echo ""
echo "✅ Download complete!"
echo ""
echo "You can now run the application with:"
echo "  python app/main.py"
