#!/bin/bash
# setup_colab.sh - Google Colab 用セットアップスクリプト

echo "🚀 SAM 3D Pose Analyzer の環境を構築中..."

# Load HF_TOKEN from .env file if it exists
if [ -f ".env" ]; then
    echo "📄 Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# 1. システムライブラリのインストール
apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    libosmesa6 libosmesa6-dev libglu1-mesa freeglut3-dev \
    blender python3-numpy

# 2. Python 依存関係のインストール
# utils3d や triton の競合回避のため、先に個別にインストールします
pip install git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183 --no-deps
pip install triton
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gdown "huggingface_hub<1.0"

# 3. 外部リポジトリのセットアップ (Git管理から外したコードを再取得)
mkdir -p repos
pushd repos
[ ! -d "sam-3d-body" ] && git clone https://github.com/facebookresearch/sam-3d-body.git
[ ! -d "sam3" ] && git clone https://github.com/facebookresearch/sam3.git
[ ! -d "MoGe" ] && git clone https://github.com/microsoft/MoGe.git
popd

# 4. モデルのダウンロード (Python スクリプトで確実に実行)
mkdir -p weights/assets
echo "📦 モデルチェックポイントを準備中..."

cat <<EOF > download_models.py
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

    os.makedirs("weights/assets", exist_ok=True)

    # SAM 3D Body models - download and copy to correct location
    print("Downloading SAM 3D Body models...")

    ckpt_path = hf_hub_download(repo_id="facebook/sam-3d-body-dinov3", filename="model.ckpt")
    shutil.copy(ckpt_path, "weights/model.ckpt")
    print("✓ Copied model.ckpt to weights/")

    mhr_path = hf_hub_download(repo_id="facebook/sam-3d-body-dinov3", filename="assets/mhr_model.pt")
    shutil.copy(mhr_path, "weights/assets/mhr_model.pt")
    print("✓ Copied mhr_model.pt to weights/assets/")

    cfg_path = hf_hub_download(repo_id="facebook/sam-3d-body-dinov3", filename="model_config.yaml")
    shutil.copy(cfg_path, "weights/model_config.yaml")
    print("✓ Copied model_config.yaml to weights/")

    print("✅ Model download complete!")

if __name__ == "__main__":
    download()
EOF

python3 download_models.py
rm download_models.py

echo "✅ セットアップ完了！"
