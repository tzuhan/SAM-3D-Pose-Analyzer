# SAM 3D Pose Analyzer

AI を活用して、一枚の画像（または動画フレーム）から **「3D ポーズ、指の動き、体型の奥行き」** を抽出し、3D キャラクターとして復元・書き出しを行うための統合ツールです。

---

## 🛠️ 事前準備 (Preparation)

本ツール（Google Colab および ローカル環境）の利用には、以下の準備が必要です。

1.  **Hugging Face アカウントの作成**: [Hugging Face](https://huggingface.co/) でアカウントを作成してください。
2.  **Access Token の取得**: [Settings -> Access Tokens](https://huggingface.co/settings/tokens) から `read` 権限のトークンを取得してください。
3.  **モデルの利用承諾**: 以下のモデルリポジトリにアクセスし、各ページの **"Agree and access repository"** ボタンを押して利用を承諾してください。
    - [facebook/sam-3d-body](https://huggingface.co/facebook/sam-3d-body) (または関連する Meta のモデルリポジトリ)
    - ※ Meta 社のモデル（SAM3 等）は、Hugging Face 上で承諾が必要な場合があります。

> [!IMPORTANT]
> **Google Colab ユーザーへ**: 起動時にトークンの入力を求められます。承諾が済んでいないと、モデルのダウンロード時にエラーが発生します。

## 📸 主な機能 (Features)

- **⚡ クイック復元 (1人専用)**: 画像を投げてボタンを押すだけで、最速で 3D 化が可能です。AI が人物を一瞬で見つけ出し、ボーンとメッシュを生成します。
- **👥 アドバンス復元 (詳細設定)**: 複数人のスキャン、特定の人物の選択、背景の奥行き（MoGe）を考慮した配置など、こだわりの設定が可能です。
- **全身の 3D 復元**: 推定された関節位置だけでなく、キャラクターのボリューム（メッシュ）を高品質に復元します。
- **クリップスタジオ (CLIP STUDIO PAINT) 対応**: 書き出した BVH データを直接読み込み、3D デッサン人形のポーズとして活用可能です。
    - ※出力された BVH をキャンバス上の 3D デッサン人形にドラッグ＆ドロップすることで即座にポーズが適用されます。
- **Unity / Unreal Engine / その他**: 3D リファレンス用アセット (OBJ, GLB 形式)

## ✨ サポートしている拡張子

- **FBX**: アニメーション用ボーン + スキニング済みメッシュ
- **BVH**: ポーズデータ
- **OBJ**: 静止メッシュデータ
- **GLB**: Web/AR 用バイナリ形式

## 🚀 実行方法 (Quick Start)

### 1. Google Colab
ブラウザだけで今すぐ試せます。

- [**SAM 3D Pose Analyzer on Colab**](https://colab.research.google.com/github/chchannel/SAM-3D-Pose-Analyzer/blob/main/sam_3d_pose_analyzer_colab.ipynb)
    1.  **Hugging Face ログイン**: セル1(Step 1)を実行し、表示されるプロンプトにトークンを入力します。
    2.  **環境構築**: セル2(Step 2)を実行します（約 10〜15 分）。
    3.  **アプリ起動**: セル3(Step 3)を実行し、発行される `public URL (gradio.live)` をクリック。

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
bash setup_colab.sh 

# アプリの起動
python app/main.py
```

## 📜 ライセンス (Licensing)

- **生成データ (Output Assets)**: 商用・非商用を問わず、**自由にご利用いただけます。**
- **ソースコード (This Repository)**: 非商用利用に限定され、無断再配布は禁止されています。
- **技術基盤**: SAM 3 D Body (Meta), SAM 3 (Meta), MoGe (Microsoft), Detectron2 (Meta) などのライセンスに準じます。

## 🤝 謝辞 (Acknowledgments / Attribution)

本ツールの開発にあたり、以下のプロジェクトのコードを利用・参考にさせていただいています。

- **BVH I/O Logic**:
    - [smpl2bvh](https://github.com/KosukeFukazawa/smpl2bvh) (MIT License) - by Kosuke Fukazawa
    - [Motion-Matching](https://github.com/orangeduck/Motion-Matching) (MIT License) - by Daniel Holden
- **Blender 3D Export Idea**:
    - [note: SAM 3D BodyのポーズをBlenderで再現する](https://note.com/tori29umai/n/n5550b2b5ec26) - by とりにく

---
*Developed by Antigravity (AI Assistant) & USER.*
