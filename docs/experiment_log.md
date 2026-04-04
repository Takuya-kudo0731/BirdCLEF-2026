# 実験記録 (Experiment Log)

各実験の設定・結果・考察を記録する。

---

## EXP-000: Perch v2 + LogisticRegression (ベースライン)

| 項目 | 値 |
|---|---|
| **日付** | 2026-03-28 |
| **Config** | なし（perch_v2_baseline.ipynb 内で設定） |
| **モデル** | Perch v2 Embedding (1536-dim) + LogisticRegression (C=0.1) |
| **損失関数** | — (sklearn) |
| **CV cmAP** | 0.8665 (+/- 0.0053) |
| **LB Score** | **0.704** |
| **GPU時間** | ~30min (Embedding抽出) |

### 考察
- CV と LB の乖離が大きい (0.87 vs 0.70)
- LogisticRegression の表現力不足
- 28種のゼロショット種が全て 0.0 → 大きな減点要因
- 音声の先頭5秒のみ使用 → 情報の取りこぼし

---

## EXP-001: EfficientNet-B3 + BCE (v1)

| 項目 | 値 |
|---|---|
| **日付** | 2026-03-28 |
| **Config** | `configs/efficientnet_v1.yaml` |
| **モデル** | tf_efficientnet_b3_ns + GeM Pooling (p=3.0) |
| **損失関数** | BCEWithLogitsLoss |
| **hop_length** | 512 (improved.yaml の 256 から変更) |
| **n_mels** | 128 (improved.yaml の 224 から変更) |
| **num_epochs** | 5 (improved.yaml の 30 から変更) |
| **CV cmAP** | — (実行待ち) |
| **LB Score** | — (実行待ち) |
| **GPU時間** | — |

### improved.yaml からの変更点
| 項目 | improved.yaml | efficientnet_v1.yaml | 変更理由 |
|---|---|---|---|
| use_focal_loss | true | **false** | 2025年上位で Focal < BCE の実績 |
| hop_length | 256 | **512** | サウンドスケープ汎化 (2025年上位) |
| n_mels | 224 | **128** | 上位解法標準、計算量削減 |
| num_epochs | 30 | **5** | 5エポック超で過学習 (2025年知見) |
| early_stopping | 7 | **3** | エポック減に合わせて |
| freq_mask_param | 27 | **16** | n_mels=128 に合わせて (~12%) |
| time_mask_param | 62 | **31** | hop_length=512 に合わせて (~10%) |

### 結果
- **Fold 0 Best cmAP: 0.7142**（5エポック全て改善、early stopping 未発動）
- **LB Score: 0.640**

### 考察
- Perch v2 (0.704) より低い。fold 0 のみ + 5エポックでは不足
- GPU 時間の制約が厳しい（1 fold ≒ 8時間）

---

## EXP-002: Perch v2 + MLP (2層NN)

| 項目 | 値 |
|---|---|
| **日付** | 2026-03-29 |
| **モデル** | Perch v2 Embedding (1536-dim) + MLP (1536→512→206) |
| **損失関数** | CrossEntropyLoss |
| **CV cmAP** | **0.8649 (+/- 0.0071)** |
| **LB Score** | — (未提出) |

### Fold別結果
| Fold | Best cmAP | Early Stop |
|------|-----------|------------|
| 0 | 0.8552 | epoch 9 |
| 1 | 0.8609 | epoch 9 |
| 2 | 0.8623 | epoch 15 |
| 3 | 0.8723 | epoch 13 |
| 4 | 0.8739 | epoch 9 |

### 考察
- CV は LogReg (0.8665) とほぼ同等 → Perch Embedding の線形分離性が高い
- MLP にしても改善されない → 分類器の変更では限界

---

## EXP-003: Perch v2 + MLP + Overlap推論

| 項目 | 値 |
|---|---|
| **日付** | 2026-03-29 |
| **モデル** | EXP-002 の MLP + 2.5秒オーバーラップ推論 |
| **LB Score** | **0.694** |

### 考察
- LogReg baseline (0.704) より **悪化**
- オーバーラップ推論は改善に寄与しなかった
- MLP の softmax 出力を平均することで確信度が薄まった可能性

---

## 全提出スコアまとめ

| # | モデル | CV cmAP | LB Score |
|---|---|---|---|
| EXP-000 | Perch v2 + LogReg | 0.8665 | **0.704** ← ベスト |
| EXP-001 | EfficientNet-B3 (fold 0) | 0.7142 | 0.640 |
| EXP-002 | Perch v2 + MLP | 0.8649 | 未提出 |
| EXP-003 | Perch v2 + MLP + Overlap | — | 0.694 |

---

<!-- 新しい実験は上に追記してください -->
<!-- テンプレート:
## EXP-XXX: タイトル

| 項目 | 値 |
|---|---|
| **日付** | YYYY-MM-DD |
| **Config** | `configs/xxx.yaml` |
| **モデル** | |
| **損失関数** | |
| **CV cmAP** | |
| **LB Score** | |
| **GPU時間** | |

### 変更点
- ...

### 考察
- ...
-->
