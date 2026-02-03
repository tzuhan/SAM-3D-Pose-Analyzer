# SAM 3D Pose Analyzer

<p align="center">
<a href="https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/"><img src='https://img.shields.io/badge/Meta_AI-Paper-4A90E2?logo=meta&logoColor=white' alt='Paper'></a>
<a href="https://ai.meta.com/blog/sam-3d/"><img src='https://img.shields.io/badge/Project_Page-Blog-9B72F0?logo=googledocs&logoColor=white' alt='Blog'></a>
<a href="https://huggingface.co/datasets/facebook/sam-3d-body-dataset"><img src='https://img.shields.io/badge/🤗_Hugging_Face-Dataset-F59500?logoColor=white' alt='Dataset'></a>
<a href="https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d"><img src='https://img.shields.io/badge/🤸_Playground-Live_Demo-E85D5D?logoColor=white' alt='Live Demo'></a>
</p>

---
![topimage](https://github.com/user-attachments/assets/f937e907-995d-44a0-945c-a447665dfcc1)


AI を活用して、一枚の画像（または動画フレーム）から **「3D ポーズ、指の動き、体型の奥行き」** を抽出し、3D キャラクターとして復元・書き出しを行うための統合ツールです。
Meta の **SAM 3D Body** をベースに、単一画像から即座に 3D リファレンスとして利用可能なアセットを出力します。

---

##  主な機能 (Features)

- **⚡ クイック復元 (1人専用)**: 画像を投げてボタンを押すだけで、最速（約60秒）で 3D 化が可能です。AI が人物を一瞬で見つけ出し、ボーンとメッシュを生成します。
- **👥 アドバンス復元 (詳細設定)**: 複数人のスキャン、特定の人物の選択、背景の奥行き（MoGe）を考慮した配置など、こだわりの設定が可能です。
- **VRMメモリ解放**:  作業ごとにVRMメモリを解放することで、バックグラウンド待機中のメモリ消費をほぼなくすことが可能です。
- **クリップスタジオ (CLIP STUDIO PAINT) 対応**: 書き出した BVH データを直接読み込み、3D デッサン人形のポーズとして活用可能です。
    - ※出力された BVH をキャンバス上の 3D デッサン人形にドラッグ＆ドロップすることで即座にポーズが適用されます。
- **様々な 3D 形式に対応**: FBX, BVH, OBJ, GLB を一括出力。Unity や Unreal Engine でリファレンスとして即座に利用可能です。

  https://github.com/user-attachments/assets/55391f3d-8e08-427b-9678-463db51d8b18



## ✨ サポートしている拡張子

- **FBX**: UNITY準拠アニメーション用ボーン + スキニング済みメッシュ
- **BVH**: ポーズデータ (CLIP STUDIO PAINT 互換)
- **OBJ**: 静止メッシュデータ
- **GLB**: Web/AR 用バイナリ形式（ブラウザ上で確認できます）

## 🚀 実行方法 (Quick Start)

本ツールを利用するには、事前に **Hugging Face でのモデル利用承諾** が必要です。

### 🛠️ 事前準備
1.  **Hugging Face アカウント作成 & トークン取得**: [Access Tokens](https://huggingface.co/settings/tokens) からトークンを取得（Read権限）。
2.  **モデルの利用承諾**: 以下のリポジトリで **"Agree and access repository"** をクリック。
    - [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3)
    - [facebook/sam3](https://huggingface.co/facebook/sam3)
3.  **環境変数の設定（ローカル環境のみ）**: `.env` ファイルにトークンを設定。
    ```bash
    cp .env.example .env
    # .env ファイルを編集して HF_TOKEN を設定
    ```

---

### 1. Google Colab 
ブラウザだけで今すぐ試せます。

- [**SAM 3D Pose Analyzer on Colab**](https://colab.research.google.com/github/chchannel/SAM-3D-Pose-Analyzer/blob/main/sam_3d_pose_analyzer_colab.ipynb)
    - **Step 1**: 実行時に Hugging Face トークンを入力します。
    - **Step 2**: 環境構築（約 10〜15 分）。
    - **Step 3**: 起動後の **⚡ クイック復元タブ** を使えば、約 60~120秒で 3D モデルが手に入ります。

### 2. ローカル環境 - Linux/WSL2 (Local Installation)
WSL2 または Linux 環境での動作を想定しています。RTX3060tiで動作確認しました。

**必要条件:**
- Python 3.10 または 3.11
- NVIDIA GPU + CUDA（推奨）

```bash
# リポジトリの取得
git clone https://github.com/chchannel/SAM-3D-Pose-Analyzer.git
cd SAM-3D-Pose-Analyzer

# システム依存ライブラリのインストール（Ubuntu/WSL2）
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    libosmesa6 libosmesa6-dev libglu1-mesa freeglut3-dev blender

# 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate

# HuggingFace トークンの設定（以下のいずれかの方法）
# 方法1: .env ファイルを使用
cp .env.example .env
nano .env  # HF_TOKEN=hf_xxxxx の形式でトークンを設定

# 方法2: HuggingFace CLI でログイン（.env の代わりに使用可能）
# pip install huggingface_hub && huggingface-cli login

# PyTorch のインストール（CUDA 12.1 版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# numpy のインストール（他のパッケージのビルドに必要）
pip install numpy

# detectron2 のインストール（PyTorch が必要なため個別にインストール）
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# chumpy のインストール（ビルド問題回避のため個別にインストール）
pip install --no-build-isolation chumpy

# その他の依存ライブラリのインストール
pip install -r requirements.txt

# 外部リポジトリとモデルのセットアップ（初回のみ）
bash setup_colab.sh

# アプリの起動
python app/main.py
```

**注意事項:**
- Ubuntu 23.04 以降では仮想環境（venv）が必須です
- 次回以降は `source .venv/bin/activate` で仮想環境を有効化してから起動してください
- NVIDIA GPU がない場合は、PyTorch を CPU 版に変更してください：
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

### 3. macOS (Apple Silicon / Intel)

macOS での実行には専用のセットアップスクリプトを使用します。

**必要条件:**
- Python 3.10 または 3.11（3.13 は非対応）
- pyenv（推奨）または Homebrew でインストールした Python 3.11

```bash
# リポジトリの取得
git clone https://github.com/chchannel/SAM-3D-Pose-Analyzer.git
cd SAM-3D-Pose-Analyzer

# Python 3.11 のインストール（pyenv を使用する場合）
brew install pyenv
pyenv install 3.11.9

# HuggingFace トークンの設定（以下のいずれかの方法）
# 方法1: .env ファイルを使用
cp .env.example .env
nano .env  # HF_TOKEN=hf_xxxxx の形式でトークンを設定

# 方法2: HuggingFace CLI でログイン（.env の代わりに使用可能）
# pip install huggingface_hub && huggingface-cli login

# macOS 用セットアップスクリプトを実行（依存関係のインストール）
./setup_macos.sh

# 外部リポジトリとモデルのダウンロード（初回のみ）
./download_models_macos.sh

# アプリの起動
source .venv/bin/activate
python app/main.py
```

**注意事項:**
- macOS では CPU または Apple MPS（Metal Performance Shaders）で実行されます
- NVIDIA GPU がないため、Colab より処理が遅くなります
- Apple Silicon (M1/M2/M3/M4) では MPS による GPU アクセラレーションが利用可能です
- `.env` ファイルは `.gitignore` に含まれているため、誤ってコミットされることはありません

## 📜 ライセンス (Licensing)

- **生成データ (Output Assets)**: 商用・非商用を問わず、**自由にご利用いただけます。**
- **ソースコード (This Repository)**: 非商用利用に限定され、無断再配布は禁止されています。
- **技術基盤**: 以下の各公式リポジトリのライセンス条件を継承します。
    - [SAM 3 D Body (Meta)](https://github.com/facebookresearch/sam-3d-body)
    - [SAM 3 (Meta)](https://github.com/facebookresearch/sam3)
    - [MoGe (Microsoft)](https://github.com/microsoft/MoGe)
    - [Detectron2 (Meta)](https://github.com/facebookresearch/detectron2)

## 🤝 謝辞 (Acknowledgments / Attribution)

本ツールの開発にあたり、以下のプロジェクトのコードを利用・参考にさせていただいています。

- **BVH I/O Logic**:
    - [smpl2bvh](https://github.com/KosukeFukazawa/smpl2bvh) (MIT License) - by Kosuke Fukazawa
    - [Motion-Matching](https://github.com/orangeduck/Motion-Matching) (MIT License) - by Daniel Holden
- **Blender 3D Export Idea**:
    - [note: SAM 3D BodyのポーズをBlenderで再現する](https://note.com/tori29umai/n/n5550b2b5ec26) - by とりにく

---
*Developed by Antigravity (AI Assistant) & USER.*





