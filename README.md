# SAM 3D Pose Analyzer

AI を活用して、一枚の画像（または動画フレーム）から **「3D ポーズ、指の動き、体型の奥行き」** を抽出し、3D キャラクターとして復元・書き出しを行うための統合ツールです。

---

## 📸 主な機能 (Features)

- **⚡ クイック復元 (1人専用)**: 画像を投げてボタンを押すだけで、最速で 3D 化が可能です。AI が人物を一瞬で見つけ出し、ボーンとメッシュを生成します。
- **👥 アドバンス復元 (詳細設定)**: 複数人のスキャン、特定の人物の選択、背景の奥行き（MoGe）を考慮した配置など、こだわりの設定が可能です。
- **全自動モード (Auto-Recovery)**: 特定の人物を選ばなくても、検出された全員を一度に 3D 化できます。
- **クリップスタジオ (CLIP STUDIO PAINT) 対応**: 書き出した BVH データを直接読み込み、3D デッサン人形のポーズとして活用可能です。
- **様々な 3D 形式に対応**: FBX, BVH, OBJ, GLB を一括出力。Unity や Unreal Engine ですぐに使えます。

## 🚀 実行方法 (Quick Start)

### 1. Google Colab
ブラウザだけで今すぐ試せます。

- [**SAM 3D Pose Analyzer on Colab**](https://colab.research.google.com/github/chchannel/SAM-3D-Pose-Analyzer/blob/main/sam_3d_pose_analyzer_colab.ipynb)
    1.  **Hugging Face ログイン**: セル1(Step 1)を実行。
    2.  **環境構築**: セル2(Step 2)を実行（約 10〜15 分）。
    3.  **アプリ起動**: セル3(Step 3)を実行し、発行される `public URL (gradio.live)` をクリック。
        - **⚡ クイック復元タブ** を使えば、約 60〜80 秒で 3D モデルが手に入ります。

### 2. ローカル環境 (Local Installation)
WSL2 または Linux 環境向けです。

```bash
git clone https://github.com/chchannel/SAM-3D-Pose-Analyzer.git
cd SAM-3D-Pose-Analyzer
pip install -r requirements.txt
bash setup_colab.sh 
python app/main.py
```

## 📜 ライセンス (Licensing / Credits)

- **生成データ (Output Assets)**: 商用・非商用を問わず、**自由にご利用いただけます。**
- **ソースコード**: 非商用利用に限定。
- **技術基盤**: SAM 3D Body (Meta), SAM 3 (Meta), MoGe (Microsoft), Detectron2 (Meta) の各成果を利用しています。
