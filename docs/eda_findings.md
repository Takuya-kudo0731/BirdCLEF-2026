# BirdCLEF+ 2026 EDA 結果と戦略への示唆

**EDA実行日:** 2026-03-28（Kaggle GPU ノートブック）
**ノートブック:** `notebooks/eda_kaggle.ipynb`（実行済み版: `notebook8b449683e6.ipynb`）

---

## データ概要

| 項目 | 値 |
|---|---|
| 総サンプル数 | **35,549件** |
| 総クラス数（訓練） | **206種** |
| 提出要求クラス数 | **234種** |
| 訓練のみの種 | 0種（すべて提出対象） |
| 提出のみの種 | **28種**（訓練データなし） |
| メタデータファイル | `train.csv`（15列） |
| 音声ディレクトリ | `train_audio/`, `train_soundscapes/`（10,658件） |

---

## クラス不均衡（最重要課題）

| 指標 | 値 |
|---|---|
| 最小サンプル数 | **1** |
| 最大サンプル数 | **499** |
| 中央値 | 125 |
| 平均 | 172.6 |
| 不均衡率（max/min） | **499倍** |
| ≤5サンプルのクラス | **18種** |
| ≤20サンプルのクラス | **36種** |

### 戦略への示唆
- 不均衡は深刻。何らかの対策が必須
- ただし **2025年上位解法では Focal Loss が BCEWithLogitsLoss に劣った実績あり**
- `improved.yaml` の `use_focal_loss: true` は再検討すること
- ≤5サンプルの18種は疑似ラベリングの優先候補

---

## データの種別（BirdCLEF+）

**鳥だけではない。**`class_name` 列に以下が含まれる：
- `Insecta`（昆虫）: 例 `Guyalna cuta`（iNat ID: 1161364）
- `Aves`（鳥類）
- その他陸生動物

BirdCLEF**+** の「+」は鳥以外の音声生物学的分類も含む意味。

---

## primary_label の形式（混在に注意）

| 形式 | 例 | 説明 |
|---|---|---|
| iNat taxon ID | `"1161364"` | 大多数。iNaturalist の種ID |
| bird code | `"sptnig1"` | 少数。iNat IDがない種はbird codeを使用 |

→ LabelEncoder の動作には影響しないが、secondary_label_map 構築時に注意。

---

## Secondary Labels

| 指標 | 値 |
|---|---|
| secondary labels 付与率 | **12.3%**（4,372 / 35,549件） |
| 合計 secondary label 数 | **7,431件** |

### 頻出 Secondary Labels（上位10種）
| 種コード | 件数 |
|---|---|
| grekis | 624 |
| whtdov | 468 |
| undtin1 | 315 |
| yecpar | 226 |
| rufhor2 | 225 |
| saffin | 183 |
| picpig2 | 172 |
| trokin | 171 |
| soulap1 | 154 |
| grasal3 | 151 |

### 戦略への示唆
- 付与率 12.3% → `secondary_label_weight: 0.5` は意味がある
- secondary labels は bird code 形式 → taxonomy.csv 経由の変換が必須（修正済み）

---

## 提出フォーマット

```
row_id: BC2026_Test_0001_S05_20250227_010002_5
→ ファイル名 + 末尾が評価秒数（5秒刻み）
```

- **5〜15秒の範囲で5秒刻み**（サンプルは3行のみ公開）
- 234列の確率値を出力

---

## ゼロショット28種問題

提出要求に含まれるが訓練データが **ゼロ** の28種：

```
'1491113', '25073', '47158son01', '47158son02', '47158son03',
'47158son04', '47158son05', '47158son06', '47158son07', '47158son08', ...
```

- `47158son01`〜`47158son08` は特定種の鳴き声タイプ別（sub-call type）
- これらは常に予測 = 0.0 → cmAP 計算時にペナルティ
- Padded cmAP（padding=5）がある程度緩和するが、対策が望ましい

### 対策候補
1. **taxonomy.csv からこれらの種の音声を特定**してゼロ以外を予測
2. **疑似ラベリング**で train_soundscapes から発見
3. 近縁種の予測スコアをコピー（リスク有り）

---

## test_soundscapes

- EDA 実行時：**0ファイル**（非公開テストセット）
- row_id の suffix から、1ファイルあたり複数の5秒ウィンドウで評価される

---

## train_soundscapes

- **10,658ファイル**（ラベル付きサウンドスケープ）
- `train_soundscapes_labels.csv` に対応ラベルあり
- 現状：訓練に未使用
- **疑似ラベリングの素材として最適**（実際の録音環境に近い）

---

## 現在の configs との差異・問題点

| 項目 | 現状(improved.yaml) | EDA/調査からの推奨 |
|---|---|---|
| 損失関数 | `use_focal_loss: true` | ⚠️ 2025年実績では BCE が優位 → 要検討 |
| hop_length | `256` | 2025年上位は `512`（サウンドスケープへの汎化） |
| secondary_label_weight | `0.5` | ✅ 12.3%付与率なので妥当 |
| 疑似ラベリング | 未実施 | train_soundscapes 10,658件を活用可 |
| ゼロショット28種 | 無対策 | 何らかの対策検討 |

---

## 優先アクション（EDA起点）

1. **損失関数の見直し** — `use_focal_loss: false` で BCEWithLogitsLoss に戻してLBを比較
2. **train_soundscapes の活用** — `train_soundscapes_labels.csv` を読み込んで訓練データに追加
3. **hop_length の変更** — `256 → 512` でサウンドスケープへの汎化を改善
4. **ゼロショット28種の調査** — `47158son01`〜`47158son08` が何の種かを taxonomy.csv で確認
