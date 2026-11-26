# 以下claudeの生成によるため誤りがある可能性があります。ご確認が必要な点がございましたらご連絡いただけると幸いです

# Pコン跡 画像認識（SDNET2018 壁面W）

256×256のタイル画像にPコン跡が含まれるか（1/0）を判定するモデルです。


最終提出：MIL系4モデルのアンサンブル、閾値0.50（Public LB 0.909）

## 1. 動作環境

| 項目 | 内容 |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA Tesla V100 32GB ×1（1モデル5foldで約25〜30分） |
| Python | 3.10 |
| ライブラリ | `pcon/requirements.txt` |

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r pcon/requirements.txt
```

**注意**：scikit-learnは1.7.2を使ってください。`StratifiedGroupKFold(shuffle=True)` の分割がバージョンで変わるため、他のバージョンでは検証スコアが一致しません。

## 2. データの配置

```
<プロジェクト>/
├── DATA_Maguire_20180517_ALL/SDNET2018/W/{CW,UW}/*.jpg   # SDNET2018（壁面Wのみ使用）
├── <配布フォルダ>/train_label.csv, test.txt, sample_submission.csv
└── pcon/                                                 # このソースコード
```

配布フォルダの場所は環境変数で指定できます（未指定時は `common.py` の既定パス）。

```bash
export PCON_DIST=/path/to/配布フォルダ       # train_label.csv, test.txt がある場所
export PCON_IMG=/path/to/SDNET2018           # W/ の親フォルダ
```

## 3. 実行手順

`pcon/` で実行します。

```bash
cd pcon

# (1) 前処理：同じ撮影写真のタイル（18列×14行）をつなぎ合わせ、1/2解像度のキャンバスを保存
python prep.py

# (2) 学習：4モデル × 5fold（写真単位のStratifiedGroupKFold）
python train2.py --name v4_mil_focal --head mil --zmin 0.4 --loss focal --alpha 0.5 --pos_frac 0.25 --epochs 30
python train2.py --name v6_lse       --head mil --zmin 0.4 --loss focal --alpha 0.5 --pos_frac 0.25 --epochs 30 --pool lse
python train2.py --name v6_filled    --head mil --zmin 0.4 --loss focal --alpha 0.5 --pos_frac 0.25 --epochs 30 --aug_filled
python train2.py --name v6_both      --head mil --zmin 0.4 --loss focal --alpha 0.5 --pos_frac 0.25 --epochs 30 --pool lse --aug_filled

# (3) 提出ファイル作成：4モデルのロジット平均 → 閾値0.50
python make_final.py      # → ../submission.csv
```

各学習は `work/oof_<name>.npy`（検証予測）と `work/test_<name>.npy`（5foldモデル平均のテスト予測）、`work/<name>_f<k>.pt`（重み）を出力します。
GPUの非決定性（cuDNN）により、再実行時に数枚の予測が変わる場合があります。

## 4. ファイル構成

| ファイル | 役割 |
|---|---|
| `common.py` | パス設定、ラベル読み込み、ファイル名→写真内位置（行・列）の変換、写真の復元 |
| `prep.py` | 写真単位のキャンバス作成（1/2解像度） |
| `train2.py` | 学習・検証・テスト予測（MILヘッド、Focal Loss、拡張） |
| `make_final.py` | 最終提出ファイル作成 |
| `ens.py`, `make_cands.py` | アンサンブル評価・提出候補の一括作成 |
| `train7.py` | 追加実験（境界セル判定、stride8、タイル切り直し、TTA、1枚判定版など） |
| `viz_errors.py`, `eval_issue.py` | 誤り分析 |
