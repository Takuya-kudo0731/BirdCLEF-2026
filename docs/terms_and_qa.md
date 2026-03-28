# 用語メモ & 質問記録

このファイルは学習の過程で出てきた用語や質問を蓄積するためのメモです。
Claude との会話で出てきた疑問点を随時追記していきます。

---

## Focal Loss

**質問日:** 2026-03-28
**質問:** 「Focal Loss, WeightedSamplerについて教えて。不均衡データを扱うんだろうけど」

**一言:** 不均衡データで「簡単なサンプルの損失を自動的に小さくする」損失関数。

**通常の BCE:**
```
Loss = -log(p)
```

**Focal Loss（Lin et al. 2017, RetinaNet）:**
```
Loss = -(1 - p)^γ * log(p)
```

- `γ`（gamma）: フォーカスパラメータ。γ=2 が一般的。
  - p=0.9（簡単）→ 損失が(0.1)^2=0.01倍に縮小
  - p=0.1（難しい）→ 損失は(0.9)^2≈0.81倍でほぼそのまま
- `α`（alpha）: クラス全体の重み。正例に 0.25〜0.75 を設定。

**本来の用途:** 物体検出（背景99% vs 前景1%）のための設計。

**BirdCLEFへの適用注意:**
- 2025年上位解法では **Focal Loss < BCEWithLogitsLoss** という結果
- 多ラベル音声分類では BCE + WeightedSampler の方が安定する傾向
- `improved.yaml` の `use_focal_loss: true` は要検討

---

## WeightedRandomSampler

**質問日:** 2026-03-28
**質問:** （Focal Loss と同時に質問）

**一言:** バッチ構成を変えることで少数クラスが均等に出現するよう補正するサンプラー。

**仕組み:**
```python
# PyTorch の WeightedRandomSampler
weights = [1 / class_count[label] for each sample]
sampler = WeightedRandomSampler(weights, num_samples=N)
DataLoader(dataset, sampler=sampler)
```

| クラス | 件数 | 重み |
|---|---|---|
| 多数クラス（499件） | 499 | 1/499 ≈ 0.002 |
| 少数クラス（1件） | 1 | 1/1.0 = 1.000 |

→ 少数クラスがバッチに多く登場する。多数クラスは「少なめに」サンプリングされる。

**Focal Loss との違い:**
- Focal Loss: 損失関数を変える（全サンプルを見る）
- WeightedSampler: バッチ構成を変える（見るサンプルの頻度を変える）
- 両者を**組み合わせることも可能**

**注意点:** 少数クラスのサンプルを繰り返し使うため、過学習しやすい。

---

## Padded cmAP（Padded class-mean Average Precision）

**質問日:** 2026-03-28（ディスカッション調査より）

**一言:** BirdCLEF 2026の評価指標。確率値のランキングで評価するため閾値チューニング不要。

**仕組み:**
1. 各クラスごとに Average Precision (AP) を計算
2. **Padding:** 各クラスのグラウンドトゥルースに 5 行の真陽性を自動付加
   → 希少種の影響を緩和
3. 全クラスの AP を平均 → cmAP

**実践的な意味:**
- 全234種に確率を出力することが重要（希少種をゼロにしない）
- 確率の品質（ランキングの正確さ）が直接スコアに影響
- F1スコアと異なり、閾値の決め方に依存しない

---

## GeM Pooling（Generalized Mean Pooling）

**質問日:** （前セッションより）

**一言:** GAP（Global Average Pooling）の一般化。パラメータ `p` で平均の「強度」を調整。

**数式:**
```
GeM(x) = (1/HW * Σ x^p)^(1/p)
```
- p=1: 通常の GAP と同じ
- p→∞: Global Max Pooling に近づく
- p=3（BirdCLEFでの設定）: 強い応答に重みを置きつつ平均の安定性も確保

**なぜ鳥の鳴き声認識に有効か:**
- 鳥の鳴き声はスペクトログラムの特定の時間・周波数帯に強い応答を持つ
- GAP は弱い応答も均等に平均してしまう
- GeM は強い応答（= 鳴き声の特徴的な部分）を強調できる

---

## SoftAUCLoss

**質問日:** 2026-03-28（ディスカッション調査より）

**一言:** AUC（Area Under the ROC Curve）を直接最適化する損失関数。2025年1位が使用。

**背景:**
- BCE はクロスエントロピーを最小化するが、AUC を直接最適化しない
- AUC は実際の評価指標と相関が高い → 直接最適化する方が有利
- "Soft" は、AUC の不連続な計算を微分可能に近似したもの

**BirdCLEFへの適用:**
- 2025年1位チームが採用し最高スコアを達成
- BCE、Focal Loss より優れた結果（2025年実績）
- 実装は `pytorch_metric_learning` などのライブラリで提供

---

## SpecAugment

**質問日:** （前セッションより）

**一言:** 音声のメルスペクトログラムに対するデータ拡張。周波数軸・時間軸をランダムにマスクする。

**種類:**
1. **Frequency Masking:** 周波数帯（mel軸）をゼロにする（縦方向）
2. **Time Masking:** 時間区間をゼロにする（横方向）

**パラメータ（improved.yaml）:**
```yaml
freq_mask_param: 27    # 最大マスク幅（mel bin数）
time_mask_param: 62    # 最大マスク幅（time frame数）
num_freq_masks: 2      # 適用回数
num_time_masks: 2      # 適用回数
```

**なぜ有効か:**
- テスト時（サウンドスケープ）には複数種が重なりノイズも多い
- 特定周波数帯や特定時間が隠れていても識別できるよう学習させる

---

## Mixup

**質問日:** （前セッションより）

**一言:** 2つのサンプルを線形補間してデータを増やす手法。

```python
x_mix = λ * x1 + (1-λ) * x2
y_mix = λ * y1 + (1-λ) * y2    # ラベルも補間
λ ~ Beta(α, α)
```

**BirdCLEFでの設定:** `mixup_alpha: 0.4`（Beta分布のパラメータ）

**注意:** ラベルが混合されるため、hard labelではなくsoft labelとの相性が良い。

---

## Noisy Student（NS）

**質問日:** （前セッションより）

**一言:** ラベルなしデータを活用した半教師あり学習の事前学習手法。`tf_efficientnet_b3_ns` の "ns" はこれ。

**手順:**
1. ラベルありデータで教師モデルを訓練
2. 教師モデルでラベルなしデータに疑似ラベルを付与
3. ノイズ（DropOut, RandAugment等）を加えながら生徒モデルを訓練
4. 2→3を繰り返す

**BirdCLEFとの関連:**
- `tf_efficientnet_b3_ns`: ImageNetのNS事前学習済み重みを使用
- BirdCLEFでも同じ発想で`pseudo_label.py`が疑似ラベリングを実装

---

## Google Perch v2

**質問日:** 2026-03-28
**質問:** 「Perch V2ってのをいろいろ調べて私に教えて」

**一言:** Google DeepMind が開発した生物音響分類モデル。154万録音・約15,000種で事前学習済み。高品質な1536次元Embeddingを提供。

**アーキテクチャ:**
- バックボーン: **EfficientNet-B3**（12Mパラメータ）— Transformerではない
- 入力: 5秒モノラル音声、32kHz（160,000サンプル）
- フロントエンド: ログメルスペクトログラム（500フレーム × 128 mel bins）
- Embedding出力: **1536次元**ベクトル
- 分類出力: 約15,000クラスのlogit

**v1 → v2 の進化:**
| | v1 | v2 |
|---|---|---|
| バックボーン | EfficientNet-B1 | EfficientNet-B3 |
| Embedding | 1280次元 | 1536次元 |
| 対象 | 鳥のみ | 鳥+昆虫+カエル+哺乳類 |
| 学習データ | Xeno-Canto のみ | XC + iNaturalist + 他（154万録音） |
| BirdSet ROC-AUC | 0.839 | 0.908 |

**学習手法:**
- 2フェーズ学習（分類器学習 → 自己蒸留）
- マルチコンポーネントMixup（複数音声混合）
- ソース予測（DIET: 5秒ウィンドウから元録音を予測する補助タスク）

**BirdCLEFでの使い方:**
```
音声 → Perch v2 → 1536次元Embedding → LogisticRegression or MLP → 予測
```
- logitをそのまま使うのは非推奨（レア種が未キャリブレーション）
- 16サンプル/クラスのfew-shotでも高性能（論文実績）
- TFLite変換で推論10倍高速化が必須（CPU 90分制約対策）
- ライブラリ: `perch-hoplite`（古い `google-research/perch` は非推奨）

**競技での位置づけ:**
- Perch v2 単体 = 強いベースライン（LB 0.908）だが上位入賞には不足
- BirdCLEF 2025の2位（0.928）はPerch不使用、カスタムCNN fine-tuningが主流
- **自前EfficientNetパイプライン + Perch v2 Embeddingのアンサンブルが最適戦略**

**論文:** "Perch 2.0: The Bittern Lesson for Bioacoustics"（arXiv, 2025年8月）
**Kaggle Models:** `google/bird-vocalization-classifier` → `perch_v2` バリアント

---
<!-- 新しい用語・質問は上に追記してください -->
