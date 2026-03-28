# 変更履歴 (Changelog)

このファイルはClaudeによる編集・調査作業の記録です。
作業終了ごとに追記します。

フォーマット:
- **日付** — セッションの日付
- **作業内容** — 何をしたか
- **変更ファイル** — 変更したファイル一覧
- **背景・理由** — なぜ変更が必要だったか

---

## 2026-03-28 — Perch v2 ベースライン構築 & 初回提出

### 作業内容
1. Google Perch v2 の調査・ドキュメント化
2. Perch v2 Embedding + LogisticRegression のベースラインノートブック作成
3. TFLite 変換パイプライン構築（Kaggle TF互換性問題の回避）
4. 初回コンペ提出

### 作成・変更ファイル
| ファイル | 内容 |
|---|---|
| `notebooks/perch_v2_baseline.ipynb` | 学習用: Embedding抽出 + CV + TFLite変換（Internet ON, GPU） |
| `notebooks/perch_v2_submit.ipynb` | 提出用: TFLite推論 + submission.csv生成（Internet OFF, CPU） |
| `docs/terms_and_qa.md` | Perch v2 の項目追加 |

### CV スコア
- **CV Mean cmAP: 0.8665** (+/- 0.0053) — LogisticRegression (C=0.1) on 1536-dim Perch v2 embeddings

### 技術的な問題と解決
1. **XlaCallModuleOp version 10 エラー**: Perch v2 SavedModel が TF 2.20+ を要求。Kaggle標準TFでは動かない
   → `pip install tensorflow>=2.20` で解決（学習ノートブック側）
2. **CUDA platform エラー**: Save & Run All で GPU なしで実行された
   → Settings で GPU T4 x2 を明示的に設定
3. **提出用ノートブックでの TF 互換性**: Internet OFF では pip install 不可
   → TFLite に変換して提出用ノートブックでは `tf.lite.Interpreter` のみ使用
4. **IndexError in CV**: 少数サンプルクラスが fold から欠落
   → `clf.classes_` を使った列マッピングで修正

### Kaggle ノートブック構成
| ノートブック | 目的 | GPU | Internet |
|---|---|---|---|
| `notebook1e05bbc851` | 学習 + TFLite変換 | GPU T4 x2 | ON |
| `notebookff23091339` | 提出 (TFLite推論) | CPU | OFF |

### 提出状態
- 初回提出済み（スコア待ち）

---

## 2026-03-28 — EDA ノートブック文字化け修正

### 作業内容
`notebooks/eda_kaggle.ipynb` のマークダウンセルのUTF-8文字化け（mojibake）を修正。
コードセルの1文字ずつ改行問題も修正。

### 変更ファイル
| ファイル | 変更内容 |
|---|---|
| `notebooks/eda_kaggle.ipynb` | 17セルのエンコーディング修正 + 2セルの行フォーマット修正 |

### 原因
ノートブックの source がlatin-1エンコードのバイト列をUTF-8として読んだ際に二重エンコーディングが発生。
`encode('latin-1').decode('utf-8')` で全セルを修復。

---

## 2026-03-28 — EDA ノートブック置き換え

### 作業内容
`notebooks/eda_kaggle.ipynb` を Kaggle 実行済みの出力付きノートブックに置き換え。

### 変更ファイル
| ファイル | 変更内容 |
|---|---|
| `notebooks/eda_kaggle.ipynb` | 空のテンプレート → Kaggle 実行済み版（876KB、グラフ・出力含む） |

### 備考
- 元ファイル: `Downloads/notebook8b449683e6.ipynb`
- このファイルは**参照用の実行済み記録**として保持
- 再実行する場合は `train_kaggle.ipynb`（修正済み）を使用すること

---

## 2026-03-28 — EDA 結果分析

### 作業内容
EDA ノートブック（`notebook8b449683e6.ipynb`）の実行結果を精読・分析し、戦略への示唆を整理。

### 作成ファイル
| ファイル | 内容 |
|---|---|
| `docs/eda_findings.md` | EDA 結果の詳細サマリーと configs との差異・推奨アクション |

### 重要発見
- **BirdCLEF+ は鳥以外も含む** — `class_name: Insecta` 等、昆虫・陸生動物も対象
- **primary_label は形式混在** — iNat ID（大多数）と bird code（`sptnig1` 等）が混在
- **28種がゼロショット** — 訓練データなし、提出要求のみ。`47158son01`〜`son08` を含む
- **⚠️ Focal Loss は要検討** — 2025年実績では BCE に劣った
- **train_soundscapes 10,658件が未活用** — `train_soundscapes_labels.csv` にラベルあり

---

## 2026-03-28 — Kaggle ディスカッション調査

### 作業内容
BirdCLEF+ 2026 の Kaggle ディスカッション・競合他社アプローチ・上位解法を調査し要約。

### 作成ファイル
| ファイル | 内容 |
|---|---|
| `docs/kaggle_discussion_summary.md` | ディスカッション調査サマリー全文 |

### 主要な発見
- **現在の LB トップ:** Perch v2（0.908）→ Google の事前学習済み基盤モデルが最強
- **⚠️ Focal Loss は 2025年の実績では BCEWithLogitsLoss に劣った** → 当プロジェクトの設定を要検討
- **推奨損失関数:** SoftAUCLoss（1位）または BCEWithLogitsLoss（2位）
- **スペクトログラム:** コーススペクトログラム（hop_length=512）がサウンドスケープへの汎化に有効
- **Silero-VAD によるデータクリーニング**（人の声除去）が有効
- **疑似ラベリング 4ラウンド反復**（Multi-Iterative Noisy Student）が 2025年1位の手法

---

## 2026-03-28 — EDA分析 & Kaggleパス/ラベル修正

### 作業内容
1. EDAノートブック (`notebook8b449683e6.ipynb`) の分析
2. Kaggle入力パスの修正 (旧フォーマット → 新フォーマット)
3. メタデータファイル名の修正 (`train_metadata.csv` → `train.csv`)
4. セカンダリラベルのエンコーディング不一致修正

### 変更ファイル
| ファイル | 変更内容 |
|---|---|
| `configs/baseline.yaml` | Kaggleパス修正、`train.csv`に修正 |
| `configs/improved.yaml` | Kaggleパス修正、`train.csv`に修正 |
| `configs/pseudo.yaml` | Kaggleパス修正、`train.csv`に修正 |
| `notebooks/train_kaggle.ipynb` | `check_data`セル・`create_config`セルのパス修正 |
| `src/dataset.py` | `BirdCLEFDataset`に`secondary_label_map`パラメータ追加、bird code → iNat ID変換ロジック追加 |
| `src/train.py` | `taxonomy.csv`読み込みと`secondary_label_map`構築、DatasetへのMap受け渡し |

### 背景・理由

**Kaggleパス変更:**
Kaggle が入力データの保存場所を変更。
- 旧: `/kaggle/input/birdclef-2026/`
- 新: `/kaggle/input/competitions/birdclef-2026/`

**ファイル名修正:**
実際のコンペデータのメタデータファイル名は `train.csv`（`train_metadata.csv` ではない）。

**セカンダリラベルのエンコーディング不一致:**
- `primary_label` 列: iNat taxon ID（例: `"1161364"`）
- `secondary_labels` 列: 6文字のbird code（例: `'grekis'`, `'whtdov'`）
- LabelEncoderはiNat IDでfitされているため、bird codeを直接 `transform()` しようとすると `ValueError` が発生
- `secondary_label_weight: 0.5` の設定が無効化されていた（silentに失敗）
- 修正: `taxonomy.csv` の `species_code → primary_label` マッピングを構築し、変換してからエンコード

### EDAからの主な発見
- クラス数: 206種（訓練）、234種（提出）→ 28種は常に予測0.0
- クラス不均衡: 最大/最小比 499倍 → Focal Lossが必須
- `train_soundscapes_labels.csv`: 10,658件のラベル付きサウンドスケープ（未活用・将来的に活用検討）
- 音声品質のratingは `4.0` と `5.0` が多数 → 高品質フィルタリングが有効な可能性

---
<!-- 以降、新しいセッションの記録を上に追加してください -->