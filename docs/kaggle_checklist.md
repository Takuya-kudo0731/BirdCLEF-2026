# Kaggle ノートブック実行チェックリスト

ノートブックをKaggleで実行する前に必ず確認する項目。

---

## 実行前チェック

### コード
- [ ] **全関数が定義済み** — `train_one_epoch`, `validate` 等が使用前に定義されているか
- [ ] **`num_workers=0`** — DataLoader の num_workers は **必ず0**（Commit モードでハングする）
- [ ] **セルの順序** — imports → config → utils → model → dataset → training loop → run の順になっているか
- [ ] **パスのハードコード回避** — `_find_input()` 等で自動検出しているか

### Kaggle 設定
- [ ] **GPU T4 x2** が選択されているか（Settings → Accelerator）
- [ ] **Internet ON** になっているか（学習用ノートブックの場合）
- [ ] **Input** に必要な Competition / Dataset が全て追加されているか

### 実行方法
- [ ] **Save & Run All (Commit)** を使う（Draft Session ではなく）
- [ ] Draft Session を使った場合は、**終了後に必ず Stop する**

## 提出用ノートブック追加チェック
- [ ] **Internet OFF**
- [ ] **Accelerator: None (CPU)** または GPU
- [ ] 外部 pip install が不要であること
- [ ] **Submit to competition** ボタンが表示されていること

## GPU 時間の目安
| 処理 | 推定時間 |
|---|---|
| EfficientNet-B3 fold 0 × 5 epochs (num_workers=0) | ~3-4時間 |
| EfficientNet-B3 全5 folds × 5 epochs | ~15-20時間 |
| Perch v2 Embedding 抽出 (35,549件) | ~30分 |
| TFLite 変換 | ~5分 |

**週の GPU 上限: 30時間**
