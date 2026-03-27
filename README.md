# BirdCLEF+ 2026

BirdCLEF+ 2026 Kaggle コンペ参加用リポジトリ。
目標: **銅メダル獲得** (pcmAP メトリクス)

---

## コンペ概要

| 項目 | 内容 |
|---|---|
| タスク | パンタナール湿地帯の野生生物音声から種を多ラベル分類 |
| 入力 | 5秒の音声セグメント (サウンドスケープから切り出し) |
| 評価指標 | **Padded cmAP** (padding_factor=5) |
| 推論制約 | **CPU のみ・90分以内** |
| 最終提出期限 | 2026-06-03 |
| エントリー期限 | 2026-05-27 |

---

## リポジトリ構造

```
BirdCLEF-2026/
├── src/
│   ├── config.py          # YAML 設定ローダー
│   ├── utils.py           # 音声読み込み、メルスペクトログラム、padded_cmap
│   ├── dataset.py         # BirdCLEFDataset, MixupDataset, CombinedDataset
│   ├── model.py           # BirdCLEFModel (timm backbone + GeM Pooling)
│   ├── train.py           # 学習スクリプト (k-fold CV)
│   ├── inference.py       # 推論スクリプト (スライドウィンドウ)
│   └── pseudo_label.py    # 疑似ラベル生成
├── configs/
│   ├── baseline.yaml      # ベースライン (EfficientNet-B0)
│   ├── improved.yaml      # 改良版 (B3-NS + GeM + Focal + Secondary Labels)
│   └── pseudo.yaml        # 疑似ラベル Round 2 用
├── notebooks/
│   ├── train_kaggle.ipynb     # Kaggle 学習ノートブック (GPU)
│   └── inference_kaggle.ipynb # Kaggle 提出用推論ノートブック (CPU)
├── scripts/
│   ├── train.sh           # ベースライン学習ショートカット
│   └── submit.sh
└── requirements.txt
```

---

## クイックスタート

### ローカル環境

```bash
pip install -r requirements.txt

# ベースライン学習 (fold 0 のみ)
python src/train.py --config configs/baseline.yaml --fold 0

# 改良版で全 fold 学習
python src/train.py --config configs/improved.yaml

# 推論 & 提出ファイル生成
python src/inference.py --config configs/improved.yaml
```

### Kaggle での提出手順

1. **学習**
   - このリポジトリを Kaggle Dataset として `birdclef-2026-code` にアップロード
   - `notebooks/train_kaggle.ipynb` を GPU ノートブックとして実行
   - 出力の `.pth` と `label_encoder.pkl` を `birdclef-2026-models` Dataset としてアップロード

2. **提出**
   - `notebooks/inference_kaggle.ipynb` を新しい CPU ノートブックとして作成
   - Datasets に `birdclef-2026` + `birdclef-2026-models` を追加
   - Internet: OFF、Accelerator: None で Submit

---

## 精度向上ロードマップ (銅メダルへの道)

### フェーズ 1: 現状のベースライン
**対象ファイル:** `configs/improved.yaml`

| 手法 | 詳細 |
|---|---|
| モデル | `tf_efficientnet_b3_ns` (Noisy Student pretrained) |
| Pooling | GeM Pooling (p=3.0, learnable) |
| 損失関数 | Focal Loss (γ=2.0, α=0.25) |
| データ拡張 | SpecAugment + Pink Noise + Mixup (α=0.4) |
| Secondary labels | soft weight=0.5 で追加学習 |
| 推論 | 2.5秒オーバーラップのスライドウィンドウ |

**期待スコア:** LB ~0.75–0.80

---

### フェーズ 2: データ拡張 & 疑似ラベル (+0.03–0.05)

```bash
# Step 1: Round 1 学習
python src/train.py --config configs/improved.yaml

# Step 2: 疑似ラベル生成
python src/pseudo_label.py --config configs/improved.yaml \
    --soundscapes_dir /kaggle/input/birdclef-2026/unlabeled_soundscapes \
    --auto_cap

# Step 3: Round 2 (疑似ラベル混合で再学習)
python src/train.py --config configs/pseudo.yaml
```

**ポイント:**
- 信頼度 ≥ 0.5 のウィンドウのみ疑似ラベルとして採用
- `per_class_cap` でクラス不均衡を防止
- Round 2 は LR を下げて過学習を抑制

---

### フェーズ 3: アーキテクチャ強化 (+0.03–0.05)

優先度順に試す:

| 手法 | 説明 | 設定例 |
|---|---|---|
| **EfficientNetV2-S** | B3-NS より大きいモデル | `model_name: tf_efficientnetv2_s` |
| **EfficientNet-B4** | B3 より 1 段階大きい | `model_name: tf_efficientnet_b4_ns` |
| **BirdSet pretrained** | 鳥の鳴き声で事前学習済み | `model_name: efficientnet_b1` + BirdSet weights |
| **Attention Pooling** | GeM の代わりに Multi-head Attention | 要 model.py 拡張 |

**`configs/improved.yaml` の `model_name` を変更するだけで試せます。**

---

### フェーズ 4: 外部データ (+0.03–0.06)

BirdCLEF 過去年度のデータを外部データとして追加:

```yaml
# configs/improved.yaml に追記
# (Kaggle Dataset として追加が必要)
extra_audio_dir: /kaggle/input/birdclef-2021-2025/train_audio
extra_metadata: /kaggle/input/birdclef-2021-2025/train_metadata.csv
```

**注意:** コンペのルールで許可された外部データのみ使用すること。

---

### フェーズ 5: アンサンブル (+0.02–0.04)

複数モデルの予測を平均するだけで安定的に改善:

```python
# 複数の configs で学習したモデルをアンサンブル
MODEL_DIRS = [
    '/kaggle/input/birdclef-2026-models-b3ns',   # EfficientNet-B3-NS
    '/kaggle/input/birdclef-2026-models-b4ns',   # EfficientNet-B4-NS
    '/kaggle/input/birdclef-2026-models-v2s',    # EfficientNetV2-S
]
# inference_kaggle.ipynb の MODELS_DIR リストを拡張
```

**銅メダルの目安:** pcmAP ~0.83–0.87 (コンペの競争状況による)

---

## 各モジュールの詳細

### `src/utils.py`
- `load_audio()`: ランダムオフセット付き音声読み込み
- `audio_to_melspec()`: dB スケール + [0,1] 正規化済みメルスペクトログラム
- `padded_cmap()`: コンペ評価指標の実装

### `src/dataset.py`
- `BirdCLEFDataset`: メインデータセット
  - `mode='train'`: ランダムオフセット + SpecAugment + Pink Noise + Secondary Labels
  - `mode='val'`: オフセット固定 + Primary Label のみ
- `MixupDataset`: Beta(alpha, alpha) Mixup
- `CombinedDataset`: 実データと疑似ラベルデータの混合

### `src/model.py`
- `GeMPooling`: 学習可能な p パラメータ付き Generalized Mean Pooling
- `BirdCLEFModel`: timm バックボーン + GeM/GAP + 線形ヘッド

### `src/train.py`
- Stratified K-Fold CV
- Focal Loss / BCEWithLogitsLoss 選択可能
- 差分学習率 (backbone: lr/10, head: lr)
- CosineAnnealingLR + Early Stopping
- AMP (FP16) 対応

### `src/inference.py`
- スライドウィンドウ推論 (overlap 対応)
- 全 fold モデルのアンサンブル
- TTA (時間反転) オプション付き

### `src/pseudo_label.py`
- 信頼度閾値フィルタリング
- クラス上限キャップ
- Secondary label 付与
- BirdCLEFDataset 互換の CSV 出力

---

## トラブルシューティング

**`librosa.load` が遅い場合:**
```bash
pip install soundfile  # 既にインストール済み
# librosa は soundfile をバックエンドとして自動使用
```

**CPU 推論が 90 分を超えそうな場合:**
- `inference_kaggle.ipynb` の `MAX_FOLD_MODELS` を 1–2 に減らす
- `OVERLAP_SECONDS` を 0 にする
- `INFERENCE_BATCH_SIZE` を増やす (メモリ使用量に注意)

**OOM (Out of Memory) エラー:**
- `batch_size` を 16 に下げる
- `num_workers` を 2 に下げる
