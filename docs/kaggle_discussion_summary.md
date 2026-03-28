# BirdCLEF+ 2026 Kaggle ディスカッション調査サマリー

**調査日:** 2026-03-28
**情報源:** Kaggle Competition Page, LifeCLEF 2026, 上位入賞者ブログ・GitHub

> Kaggle のディスカッションページは JS レンダリングのため直接取得不可。
> 最新スレッドはブラウザで https://www.kaggle.com/competitions/birdclef-2026/discussion を確認。

---

## 競技基本情報

| 項目 | 内容 |
|---|---|
| 開始日 | 2026-03-11 |
| エントリー締切 | 2026-05-27 |
| 最終提出締切 | 2026-06-03 |
| 賞金 | $50,000 |
| 参加者数 | 約 3,659人エントリー、355人参加、340チーム（3/28時点） |
| 提出件数 | 約 1,528件（3/28時点） |
| 主催 | Cornell Lab of Ornithology + TU Chemnitz + Google DeepMind |

---

## 評価指標：Padded cmAP

- **閾値不要**のランキングベース指標（F1スコアのように確率を二値化不要）
- **Padding:** 各種に対し自動的に **5行の真陽性** を付加することで、希少種の影響を抑制
- F1スコアから変更された理由：F1では閾値の取り方に自由度がありモデルの品質評価が難しかったため
- **実践的意味:** 全種に均等に確率を出力することが重要。ゼロサンプル種への予測も加点対象

---

## LBスコア状況（2026年3月28日時点）

| ノートブック | 作者 | LBスコア |
|---|---|---|
| Perch v2 推論 | yashanathaniel | **0.908** |
| SED Baseline | aidensong123 | 0.862 |
| Starter Notebook | baidalinadilzhan | 0.792 |

---

## 有力アプローチ

### A. Google Perch v2（現時点でトップスコア: 0.908）
- Google DeepMind が 2025年8月にリリースしたバイオアコースティクス基盤モデル
- 鳥類・陸生動物の何百万件もの録音で事前訓練済み
- Kaggle Models で公開: https://www.kaggle.com/models/google/bird-vocalization-classifier
- → **まずこのモデルをフォークして試すのが最速**

### B. SED（Sound Event Detection）ベースライン（LB: 0.862）
- https://www.kaggle.com/code/aidensong123/birdclef-2026-sed-baseline-lb-0-862

### C. EfficientNet + Mel Spectrogram（王道手法）
- BirdCLEF 2025 の上位解法で実証済み
- 音声 → 5秒チャンク → Mel Spectrogram → 画像分類問題
- **推奨スペクトログラムパラメータ（2025上位）:**
  - コーススペクトログラム: `n_fft=2048, hop_length=512, n_mels=128`（サウンドスケープへの汎化が良い）
  - 細粒: `n_fft=1024, hop_length=64, n_mels=148`
- GeM Pooling を Layer3 + Layer4 の両方に適用

---

## BirdCLEF 2025 上位解法（2026への転用価値高）

### 1位: Nikita Babych（LB ~0.933）
- **手法:** EfficientNet アンサンブル（4×v2_s + 3×v2_b3 + 4×b3_ns + 2×b0_ns）
- **損失関数:** SoftAUCLoss（Focal Loss より優れた）
- **疑似ラベリング:** 4ラウンド反復（Multi-Iterative Noisy Student）
- **外部データ:** Xeno-Canto + iNaturalist（鳥: 5,489件 + 昆虫・両生類: 17,197件）
- **データクリーニング:** 希少クラス（< 30サンプル）は手動でノイズ除去

### 2位: VSydorskyy（LB 0.928）
- **モデル:** eca_nfnet_l0 + tf_efficientnetv2_s_in21k
- **手法:** 2021〜2024年の BirdCLEF データで事前訓練 → 2025 データで Fine-tune
- **半教師あり学習:** 蒸留（Distillation）ベース
- GitHub: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place

### Top 2% (38位): Max Melichov（LB 0.902）
- **モデル:** EfficientNet-B0 + Quantile-Mix アンサンブル + SED モデル融合
- **データクリーニング:** Silero-VAD で訓練音声から人の声を除去（ハルシネーション防止）
- **ポイント:** シンプルな手法の組み合わせが複雑な手法を上回った

---

## 効果があった手法 vs なかった手法

### ✅ 効果あり（2025実績）
- **SoftAUCLoss**（Focal Loss・BCEより優れた）
- **Quantile-Mix アンサンブル**（α=0.5: 平均 + ランク平均の組み合わせ）
- **コーススペクトログラム**（高 hop_length がサウンドスケープへの汎化に有効）
- **Silero-VAD によるデータクリーニング**（人の声除去）
- **疑似ラベリング（4ラウンド反復）**
- **過去年度 BirdCLEF データでの事前訓練**
- **CNNモデル + SEDモデルの組み合わせ**
- GeM Pooling（Layer3 + Layer4）

### ❌ 効果なし（2025実績）
- **Focal Loss**（BCEWithLogitsLoss より劣った）← **当プロジェクトで使用中！要検討**
- CutMix
- 5エポック超のトレーニング（過学習）
- エネルギーベースのセグメント選択（RMS最大部分 < 中間5秒固定）
- 2.5D CNN（複数チャネル Mel）
- Stratified K-Fold

---

## 外部データ使用ルール（推測）

**要確認:** https://www.kaggle.com/competitions/birdclef-2026/rules

- **2025年の実績:** Xeno-Canto + iNaturalist が使用された（上位入賞者複数が採用）
- 過去のルール: 外部データは「再現可能」「参照元明記」「テストデータを含まない」が条件
- 少なくとも1件の外部データなし提出が必要（過去ルール）
- 推論時はインターネット接続不可（オフライン制約）

---

## 推奨アクション（優先度順）

1. **Perch v2 ベースラインをフォーク**してまず LB スコアを確認（0.908）
2. **Focal Loss → BCEWithLogitsLoss / SoftAUCLoss に変更検討**（2025年の実績より）
3. **コーススペクトログラムパラメータに変更**（hop_length 256 → 512 検討）
4. **Unlabeled soundscapes への疑似ラベリング**（2025年 1位・2位が採用）
5. **Silero-VAD によるデータクリーニング**（人の声除去）
6. **アンサンブル**: EfficientNet + Perch/SED を Quantile-Mix で結合
7. **過去年度データ（2021〜2024）での事前訓練**

---

## 重要リンク

| リソース | URL |
|---|---|
| 競技ページ | https://www.kaggle.com/competitions/birdclef-2026 |
| ディスカッション | https://www.kaggle.com/competitions/birdclef-2026/discussion |
| ルール | https://www.kaggle.com/competitions/birdclef-2026/rules |
| Perch v2 ノートブック | https://www.kaggle.com/code/yashanathaniel/birdclef-2026-perch-v2-0-908 |
| SED Baseline | https://www.kaggle.com/code/aidensong123/birdclef-2026-sed-baseline-lb-0-862 |
| Google Perch モデル | https://www.kaggle.com/models/google/bird-vocalization-classifier |
| 2025年2位 GitHub | https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place |
| LifeCLEF 2026 公式 | https://www.imageclef.org/BirdCLEF2026 |
