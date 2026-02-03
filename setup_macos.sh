#!/bin/bash
# ===========================================
# SAM 3D Pose Analyzer - macOS Setup Script
# ===========================================

set -e  # Exit on error

echo "=========================================="
echo "SAM 3D Pose Analyzer - macOS Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Initialize pyenv if available
if command -v pyenv &> /dev/null; then
    echo -e "${GREEN}✓ Found pyenv${NC}"
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"

    # Check if Python 3.11 is installed via pyenv
    if pyenv versions | grep -q "3.11"; then
        # Set Python 3.11 for this project
        pyenv local 3.11.9 2>/dev/null || pyenv local 3.11
        PYTHON_CMD="python"
        echo -e "${GREEN}✓ Using pyenv Python 3.11${NC}"
    else
        echo -e "${RED}✗ Python 3.11 not found in pyenv${NC}"
        echo ""
        echo "Install it with:"
        echo "  pyenv install 3.11.9"
        exit 1
    fi
else
    # Fallback to system Python 3.10 or 3.11
    PYTHON_CMD=""
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        echo -e "${GREEN}✓ Found Python 3.11${NC}"
    elif command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
        echo -e "${GREEN}✓ Found Python 3.10${NC}"
    else
        echo -e "${RED}✗ Python 3.10 or 3.11 not found${NC}"
        echo ""
        echo "Please install Python 3.11 using pyenv:"
        echo "  brew install pyenv"
        echo "  pyenv install 3.11.9"
        echo ""
        echo "Or install directly:"
        echo "  brew install python@3.11"
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python $PYTHON_VERSION"

# Create virtual environment
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}! Virtual environment already exists at $VENV_DIR${NC}"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created at $VENV_DIR${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (required for detectron2)
echo ""
echo "=========================================="
echo "Step 1: Installing PyTorch..."
echo "=========================================="
pip install torch torchvision torchaudio

# Verify PyTorch installation
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
python -c "import torch; print(f'MPS (Apple Silicon GPU) available: {torch.backends.mps.is_available()}')"

# Install other requirements
echo ""
echo "=========================================="
echo "Step 2: Installing other dependencies..."
echo "=========================================="
pip install -r requirements_macos.txt

# Install detectron2 separately (requires torch to be installed first)
echo ""
echo "=========================================="
echo "Step 3: Installing detectron2..."
echo "=========================================="
echo "This may take a few minutes to compile..."

# Install build dependencies first
pip install wheel setuptools

# --no-build-isolation: use installed torch instead of creating isolated build env
CC=clang CXX=clang++ pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# Verify detectron2 installation
python -c "import detectron2; print(f'detectron2 {detectron2.__version__} installed successfully')" || echo -e "${YELLOW}Warning: detectron2 import failed, but installation may still work${NC}"

# Setup HuggingFace authentication
echo ""
echo "=========================================="
echo "Step 4: HuggingFace Authentication..."
echo "=========================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}! No .env file found${NC}"
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file${NC}"
    echo ""
    echo "Please edit .env and add your HuggingFace token:"
    echo "  1. Get your token from: https://huggingface.co/settings/tokens"
    echo "  2. Accept the model licenses:"
    echo "     - https://huggingface.co/facebook/sam-3d-body-dinov3"
    echo "     - https://huggingface.co/facebook/sam3"
    echo "  3. Edit .env and replace 'hf_xxxxxxxxx' with your actual token"
    echo ""
    read -p "Have you added your HuggingFace token to .env? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${YELLOW}Skipping model download.${NC}"
        echo "After adding your token to .env, run:"
        echo "  source .venv/bin/activate && bash setup_colab.sh"
        SKIP_DOWNLOAD=1
    fi
else
    echo -e "${GREEN}✓ Found .env file${NC}"
    # Load HF_TOKEN from .env
    export $(grep -v '^#' .env | xargs)
    if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" ]; then
        echo -e "${YELLOW}! HF_TOKEN not set or still using placeholder${NC}"
        echo "Please edit .env and add your actual HuggingFace token"
        SKIP_DOWNLOAD=1
    fi
fi

# Run the model download setup (use macOS-specific script)
if [ -z "$SKIP_DOWNLOAD" ] && [ -f "download_models_macos.sh" ]; then
    echo ""
    echo "=========================================="
    echo "Step 5: Downloading models..."
    echo "=========================================="
    bash download_models_macos.sh
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Setup complete!${NC}"
echo "=========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
if command -v pyenv &> /dev/null; then
echo "Note: pyenv is configured for this project (.python-version file created)"
echo "      Python 3.11 will be used automatically in this directory."
echo ""
fi
echo "To start the application, run:"
echo "  python app/main.py"
echo ""
