# 改善アイデアまとめ

Discussion や調査から得られた改善アイデアを記録する。

---

## 1. 人間の音声除去（Silero-VAD）

**情報源:** Discussion + 2025年 Top 2% (Max Melichov) の実績

**問題:** 訓練音声に人間の発声（解説、録音者の声）が含まれている。
モデルがこれをノイズとして学習し、テスト時のサウンドスケープとのドメインギャップが拡大。

**対策:** Silero-VAD (Voice Activity Detector) で人の声区間を検出→除去
- Silero-VAD: https://github.com/snakers4/silero-vad
- PyTorch ベース、CPU で高速動作
- 2025年 Top 2% が採用し効果を確認

**実装:**
```python
import torch
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
(get_speech_timestamps, _, _, _, _) = utils

# 音声から人の声の区間を検出
speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=32000)
# 人の声の区間をゼロに置換 or 除去して学習に使用
```

**期待効果:** 2025年 Top 2% で確認済み
**難易度:** 低（前処理として追加するだけ）
**GPU:** 不要（CPU で動作）

---

## 2. 過去年度 BirdCLEF データでの事前学習

**情報源:** Discussion + 2025年上位解法

**概要:** BirdCLEF 2021〜2025 のデータを使って事前学習し、2026 データで Fine-tune する。
種の重複がある年度のデータは特に有効。

**2025年上位の実績:**
- 2位 (VSydorskyy): 2021〜2024 で事前学習 → 2025 で Fine-tune → LB 0.928
- Top 2% (Max Melichov): 過去データ事前学習で 0.855 → 0.868（+0.013）
- 「Xeno-canto追加データより過去コンペデータの方が効果が高い」との報告

**過去コンペデータ（Kaggle で公開）:**
| コンペ | 年 | 種数 | URL |
|---|---|---|---|
| BirdCLEF 2021 | 2021 | 397種 | kaggle.com/c/birdclef-2021 |
| BirdCLEF 2022 | 2022 | 152種 | kaggle.com/c/birdclef-2022 |
| BirdCLEF 2023 | 2023 | 264種 | kaggle.com/c/birdclef-2023 |
| BirdCLEF 2024 | 2024 | 182種 | kaggle.com/c/birdclef-2024 |
| BirdCLEF 2025 | 2025 | 206種 | kaggle.com/c/birdclef-2025 |

**実装方針:**
1. 過去コンペの train_audio + train.csv をダウンロード
2. 2026年の種と重複する種のデータを抽出
3. 全データで事前学習（EfficientNet or SED）
4. 2026年データで Fine-tune

**期待効果:** +0.01〜0.03
**難易度:** 中（データダウンロードとパス整理が主な作業）
**GPU:** 必要（事前学習に時間がかかる）

---

## 3. SED ベースライン（LB 0.862）

**情報源:** 公開ノートブック

**概要:** EfficientNet + AttBlock（時間方向 Attention）で音声イベント検出。
時間軸を保持し、鳴き声の瞬間だけに注目して予測する。

**公開ノートブック:** https://www.kaggle.com/code/aidensong123/birdclef-2026-sed-baseline-lb-0-862

**対応:** Copy & Edit でフォーク → そのまま Submit で LB 0.862 が出る見込み

---

## 4. Xeno-canto 外部データ

**情報源:** Discussion + 2025年上位解法

**概要:** Xeno-canto（野鳥音声DB）から追加の訓練データをダウンロード。

**優先度:** 低（ベースが 0.85+ になってから効果が出る）

---

## 5. 複数窓の集約方法改善（最大値）

**概要:** テスト音声の推論時、複数5秒窓の予測を平均ではなく最大値で集約。

**状況:** overlap推論（平均集約）は悪化した (0.704→0.694)。最大値集約は未検証。

---

## 優先度順

| # | 施策 | 期待効果 | 難易度 | GPU |
|---|---|---|---|---|
| 1 | **SED ベースラインをフォーク** | LB 0.862 | 低 | Submit時のみ |
| 2 | **Silero-VAD で人の声除去** | +0.01〜0.03 | 低 | 不要 |
| 3 | **過去年度データで事前学習** | +0.01〜0.03 | 中 | 必要 |
| 4 | 最大値集約 | +0.02〜0.05 | 低 | 不要 |
| 5 | Xeno-canto 外部データ | +0.01〜0.03 | 高 | 必要 |
