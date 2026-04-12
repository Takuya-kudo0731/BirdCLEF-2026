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

### 考察
- Fold 0: CV cmAP 0.7142, LB 0.640
- CV-LB 乖離が EXP-000 と同程度

---

## EXP-002: Perch v2 + MLP

| 項目 | 値 |
|---|---|
| **日付** | 2026-03-30 |
| **モデル** | Perch v2 Embedding (1536-dim) + MLP (512, BN, ReLU, Dropout0.3) |
| **CV cmAP** | 0.8649 (+/- 0.0071) |
| **LB Score** | 0.694 |

### 考察
- LogReg とほぼ同等の CV、LB は若干下がった
- embedding ベースの限界か

---

## EXP-003: Perch v2 Logit Direct

| 項目 | 値 |
|---|---|
| **日付** | 2026-04-02 |
| **モデル** | Perch v2 logits (14,795-dim) → species mapping → sigmoid |
| **CV cmAP** | — (validation Top-10 accuracy 95%) |
| **LB Score** | **0.719** |

### 考察
- 学習なしで LB 0.719、これまでの最高スコア
- 203/234 種を Perch logit にマッピング

---

## EXP-004: SED (EfficientNet-B0 + AttentionSEDHead) — Fold 0

| 項目 | 値 |
|---|---|
| **日付** | 2026-04-04 |
| **モデル** | tf_efficientnet_b0.ns_jft_in1k + GEMFreqPool + AttentionSEDHead |
| **損失関数** | BCEWithLogitsLoss (clipwise + max(framewise)) |
| **num_epochs** | 10 (early stopping 3, 発動せず) |
| **batch_size** | 32 |
| **n_mels** | 256 |
| **hop_length** | 512 |
| **CV cmAP** | **0.8678** |
| **LB Score** | 0.687 (Fold0) / 0.708 (全データ) / 0.743 (row_id修正) / **0.748** (アップサンプリング) |
| **GPU時間** | ~4.2時間 (Fold0) / ~5.5時間 (全データ) |

### トラブル記録
1. v1: VAD + 毎バッチ OGG デコード → 1epoch 5時間、完走不可能で中止
2. v2: VAD + npy キャッシュ → ディスク容量不足 (No space left on device) で中止
3. v3: VAD なし + 直接 OGG 読み込み → 1epoch ~30分、10epoch 完走

### 考察
- CV 0.8678 は Perch v2 LogReg (0.8665) とほぼ同等
- row_id が start_sec (0,5,10...) だったのを end_sec (5,10,15...) に修正 → 0.708→0.743 に大幅改善
- Perch v2 Logit Direct (0.719) を超えた
- 次の改善: MixUp, SpecAugment, アップサンプリング, ONNX化

---

## EXP-005: aidensong123 SED Baseline (学習済みチェックポイント流用)

| 項目 | 値 |
|---|---|
| **日付** | 2026-04-07 |
| **モデル** | aidensong123 の best_fold0.pt (EfficientNet-B0 + SED) |
| **学習** | なし（公開チェックポイントをそのまま使用） |
| **LB Score** | **0.830** |

### 考察
- 公開ノートブックは best_fold1.pt で LB 0.862 だが、bestfold データセットには best_fold0.pt のみ
- fold0 と fold1 の差で 0.862→0.830 になった可能性
- ピーク正規化、norm="slaney"、バッチ推論、ThreadPoolExecutor など推論の工夫あり
- 自前モデル(0.748)より大幅に高い → 学習データや学習手法の差が大きい
- final_fold0.pt: LB 0.827 → best_fold0.pt (0.830) の方が良い

---

## EXP-006: SED EfficientNet-B3 学習 (aidensong123学習コード流用)

| 項目 | 値 |
|---|---|
| **日付** | 2026-04-07 |
| **モデル** | tf_efficientnet_b3.ns_jft_in1k + SED |
| **学習コード** | aidensong123公開の学習ノートブックを流用 |
| **改善点** | B0→B3, MixUp, SpecAugment, AudioAug, secondary_labels, soundscape segments |
| **LB Score** | **0.872** |
| **CV (val_auc)** | 0.9524 |
| **GPU時間** | ~7.2時間 (P100) |

### トラブル記録
1. Internet OFF のまま実行 → timmがHuggingFaceから重みをDLできずエラー
   - **対策**: 学習時は必ず Internet ON にする（pretrained=True の場合）

### 追加実験
- B3 + 10秒チャンク (Colab学習): CV 0.9622, **LB 0.857**
  - CVは5秒版(0.9524)より良いがLBは5秒版(0.872)より悪い
  - テスト音声が5秒単位評価のため、10秒学習は不利な可能性

### 追加実験2
- B3 + 周波数方向Attention (メルスペクトログラム転置): CV 0.9521, **LB 0.877**
  - 通常版B3 (0.872) より +0.005 改善
  - 「どの周波数帯で鳴いたか」を判定するモデル

### Kaggle実行チェックリスト (再発防止)
- [ ] GPU: **T4 x2** (P100はCUDA互換性エラーが出る場合あり。T4が安全)
- [ ] Internet: **ON** (学習時、pretrained=True の場合)
- [ ] Internet: **OFF** (推論/提出時)
- [ ] num_workers: 0 (Commit デッドロック防止)
- [ ] Input: コンペデータ追加済み
- [ ] output_dir: Colabなら `/content/drive/MyDrive/...`、Kaggleなら `/kaggle/working`

### 追加実験3 (マルチウィンドウ推論)
- B3 freq-attention + マルチウィンドウ推論 (通常窓+2.5秒ずらし窓の平均): **LB 0.890**
  - freq-attention単体(0.877)から +0.013 改善
  - モデル変更なし、推論方法だけで改善
  - GPU不要、提出ノートブックの修正のみ

### 追加実験4
- B3 + freq-attention + 過去年データ (Kaggle学習、7エポックで打ち切り): **LB 0.866**
  - freq-attention単体(0.877)より下がった
  - 原因候補: エポック不足 / 過去年データのドメイン差 / val汚染

### エラー履歴
1. Internet OFF で学習実行 → timmがHuggingFaceから重みをDLできずエラー
2. GPU P100 で MixUp 実行 → `CUDA error: no kernel image is available` → **T4 x2 に変更で解決**
3. Colab で output_dir が `/kaggle/working` のまま → チェックポイント消失 → Colab対応の自動切替で解決

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
