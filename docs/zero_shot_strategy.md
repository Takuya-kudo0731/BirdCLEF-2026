# ゼロショット28種 対策検討

## 問題の整理

- 提出要求: **234種**
- 訓練データ: **206種**
- ゼロショット: **28種**（訓練データなし、常に 0.0 を出力中）
- 現在のベストLB: **0.704**（Perch v2 + LogReg）

### 28種の内訳（2026-04-03 確認済み）

| 分類 | 種数 | 内容 | train_soundscapes |
|---|---|---|---|
| 昆虫ソノタイプ `47158son01`〜`son25` | **25種** | 未同定昆虫の音響パターン25タイプ（Insecta） | **なし** |
| 両生類 `1491113` (Adenomera guarani) | 1種 | カエル（Amphibia） | **あり（14行）** |
| 両生類 `25073` (Chiasmocleis mehelyi) | 1種 | カエル（Amphibia） | **なし** |
| 両生類 `517063` (不明) | 1種 | 要確認 | **あり（36行）** |

**重要**: ソノタイプは `son08` ではなく **`son25` まで（25種）** あった。外部DBからの取得は不可能。

### train_soundscapes_labels.csv の構造
```
filename, start, end, primary_label
BC2026_Train_0039_..., 00:00:00, 00:00:05, 22961;23158;24321;517063;65380
```
- 5秒ごとのセグメントにラベル付き
- `primary_label` は**セミコロン区切りの複数ラベル**
- 1,478行、251種

---

## 対策案（優先度順）

### 1. train_soundscapes_labels.csv の確認 ★最優先

**期待効果: 高 / 難易度: 低 / GPU: 不要**

train_soundscapes（10,658件）のラベルデータに28種が含まれている可能性が高い。
なぜなら、ソノタイプはパンタナール湿地の録音から定義されたもので、train_soundscapes はまさにその環境の録音だから。

```python
ts_labels = pd.read_csv('train_soundscapes_labels.csv')
zero_shot_species = sub_species - train_species
found = zero_shot_species & set(ts_labels['primary_label'].unique())
print(f"発見: {len(found)} / 28種")
```

**もし見つかれば**: 該当セグメントの音声を切り出し → Perch v2 Embedding → 分類器の訓練データに追加。

### 2. taxonomy.csv の詳細分析

**期待効果: 中 / 難易度: 低 / GPU: 不要**

28種それぞれの学名・分類情報を特定する。種が同定できれば外部データ取得が可能。

```python
tax = pd.read_csv('taxonomy.csv')
for sp in zero_shot_species:
    row = tax[tax['primary_label'] == sp]
    if len(row) > 0:
        print(f"{sp}: {row.iloc[0].to_dict()}")
```

### 3. Perch v2 logit 出力の活用

**期待効果: 中 / 難易度: 中 / GPU: 不要（推論時TFLiteで可）**

Perch v2 は Embedding (1536-dim) の他に **~15,000種の logit** も出力する。
28種に対応する logit があれば、直接スコアとして使える。

**注意点:**
- logit は未キャリブレーション → 信頼性が低い（論文で明記）
- ソノタイプ（47158son01〜08）は Perch の labels.csv に存在しない可能性が高い
- 種が同定できているものは対応する logit index を使える可能性あり

### 4. 外部データ取得（ソノタイプ以外の20種向け）

**期待効果: 高 / 難易度: 中 / GPU: Embedding抽出に必要**

taxonomy.csv で学名が判明した種について、Xeno-Canto / iNaturalist API で音声データを取得。

- 2025年1位: Xeno-Canto 5,489件 + iNaturalist 17,197件を追加
- **ルール**: 外部データ使用可（再現可能・参照元明記・テストデータ非含有）
- **制約**: 推論時 Internet OFF → 事前にダウンロードして Dataset として追加

### 5. 疑似ラベリング（全28種対応可能）

**期待効果: 高 / 難易度: 高 / GPU: 必要**

#### (a) Perch v2 logit ベース
1. 全 train_soundscapes を Perch v2 で推論
2. 28種対応の logit を確認、高スコアセグメントを候補として抽出
3. 候補を訓練データに追加

#### (b) クラスタリングベース（ソノタイプ向け）
1. 全 train_soundscapes の Embedding を抽出
2. 既知206種の centroid から遠いクラスターを発見
3. これらを未知ソノタイプの候補として分析

#### (c) 反復疑似ラベリング（2025年1位手法）
1. 既知206種で強いモデル訓練
2. train_soundscapes で推論 → 高信頼予測を訓練データに追加
3. 再訓練 → 再推論を4ラウンド反復
4. 2025年1位: 0.87 → 0.93 まで改善

### 6. 近縁種スコアのコピー（リスク有り）

**期待効果: 小 / 難易度: 低 / GPU: 不要**

taxonomy.csv で同一科・同一属の種を特定し、その予測スコアをゼロショット種にコピー。
音響特性が大きく異なる場合はノイズになるリスクあり。

---

## Padded cmAP への影響試算

- 28種 / 234種 = **12%** のクラスが無対策
- padding_factor=5 により完全0ではないが、正の予測がなければ AP は低い
- 全体 cmAP への影響: **-0.02〜0.05** 程度
- 銅メダル圏（~0.85）を狙うには対策が必要

---

## 推奨実行計画

| Step | アクション | タイミング | GPU |
|------|-----------|----------|-----|
| **1** | train_soundscapes_labels.csv を分析 | **今すぐ**（Kaggle CPU ノートブックで） | 不要 |
| **2** | taxonomy.csv から28種の情報を特定 | **今すぐ** | 不要 |
| **3** | 28種が train_soundscapes に含まれていれば → 該当音声で Embedding 抽出 + 分類器再学習 | GPU リセット後 | 必要 |
| **4** | 含まれていなければ → Perch logit 活用 or 外部データ取得 | GPU リセット後 | 必要 |

**Step 1, 2 は GPU 不要なので今すぐ実行可能。**

---

## 参考: 2025年上位解法のゼロショット関連手法

| 順位 | 手法 | 効果 |
|------|------|------|
| 1位 | 外部データ + 4ラウンド疑似ラベリング | 0.87 → 0.93 |
| 2位 | 過去年度データ事前学習 + 外部データ + 自己蒸留 | 0.928 |
| Top 2% | Silero-VAD + train_soundscapes 疑似ラベリング | 0.902 |
