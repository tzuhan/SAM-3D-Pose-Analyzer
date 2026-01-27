# SAM 3D Pose Analyzer

AI を活用して、一枚の画像（または動画フレーム）から **「3D ポーズ、指の動き、体型の奥行き」** を抽出し、3D キャラクターとして復元・書き出しを行うための統合ツールです。

---


### 2. ローカル環境 (Local Installation)
WSL2 または Linux 環境での動作を想定しています。

リポジトリを軽量化しているため、初回実行前に外部リポジトリとモデルの取得が必要です。
```bash
# リポジトリの取得
git clone https://github.com/chchannel/SAM-3D-Pose-Analyzer.git
cd SAM-3D-Pose-Analyzer

# 依存ライブラリのインストール
pip install -r requirements.txt

# 外部リポジトリとモデルのセットアップ（初回のみ）
# 注意: Blender がインストールされている必要があります
bash setup_colab.sh 

# アプリの起動
python app/main.py
```
> [!NOTE]
> `setup_colab.sh` は Colab 用ですが、WSL2/Linux 環境でも外部リポジトリの取得やモデルのダウンロードに利用可能です。

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
    - [note: SAM 3D BodyのポーズをBlenderで再現する](https://note.com/tori29umai/n/n5550b2b5ec26) - by とり

---
*Developed by Antigravity (AI Assistant) & USER.*
