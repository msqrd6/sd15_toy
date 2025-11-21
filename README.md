# SD15_Toy

Stable Diffusion 1.5の学習・推論を簡潔に行うためのPyTorchベースのツールキットです。LoRAとControlNetの学習・推論に対応しています。

## 特徴

- **シンプルな実装**: 最小限の依存関係で、SD1.5の学習・推論を実現
- **LoRA対応**: UNetとTextEncoderへのLoRA注入と学習
- **ControlNet対応**: マルチControlNetによるガイダンス付き生成
- **Accelerate統合**: 分散学習とmixed precisionに対応
- **柔軟なデータセット**: カスタムデータセットで簡単に学習可能

## 必要要件

- Python 3.8+
- CUDA対応GPU
- PyTorch 2.0+
- Diffusers
- Transformers
- Accelerate

## インストール

```bash
git clone https://github.com/msqrd6/sd15_toy.git
cd sd15_toy
pip install -r requirements.txt
```

## プロジェクト構成

```
sd15_toy/
├── eval.py              # 基本的な推論スクリプト
├── lora_eval.py         # LoRA推論スクリプト
├── lora_train.py        # LoRA学習スクリプト
├── cn_eval.py           # ControlNet推論スクリプト
├── cn_train.py          # ControlNet学習スクリプト
├── dataset/             # 学習データセット用ディレクトリ
└── utils/               # ユーティリティモジュール
    ├── utils.py         # 汎用ユーティリティ
    ├── lora_utils.py    # LoRA関連ユーティリティ
    ├── dataset_utils.py # データセット処理
    ├── training_manager.py  # 学習進捗管理
    └── convert_utils.py # モデル変換ユーティリティ
```

## 使い方

### 1. 基本的な推論 (`eval.py`)

Stable Diffusion 1.5を使った基本的な画像生成を行います。

```python
# eval.py内のパラメータを設定
model_path = "path/to/your/sd15/model"  # diffusers形式
prompts = "a beautiful landscape"
negative_prompt = "low quality, blurry"
```

```bash
python eval.py
```

**主要パラメータ:**
- `model_path`: diffusers形式のSD1.5モデルパス
- `sampling_steps`: サンプリングステップ数（デフォルト: 20）
- `guidance_scale`: CFGスケール（デフォルト: 7.5）
- `width`, `height`: 生成画像サイズ（デフォルト: 512x512）

### 2. LoRA推論 (`lora_eval.py`)

学習済みLoRAを読み込んで画像生成を行います。

```python
# lora_eval.py内のパラメータを設定
model_path = "path/to/your/sd15/model"
lora_path = "path/to/your/lora.safetensors"
```

```bash
python lora_eval.py
```

**注意事項:**
- LoRAのstate_dictのキー形式によっては、正しく注入できない場合があります
- 互換性のあるLoRA形式を使用してください

### 3. LoRA学習 (`lora_train.py`)

UNetとTextEncoderにLoRAを注入して学習を行います。

```python
# lora_train.py内のパラメータを設定
model_path = "path/to/your/sd15/model"
dataset_path = "dataset"
rank = 128
alpha = 64
num_epochs = 20
batch_size = 5
```

```bash
accelerate launch lora_train.py
```

**主要パラメータ:**
- `rank`: LoRAのランク（デフォルト: 128）
- `alpha`: LoRAのアルファ値（デフォルト: 64）
- `num_epochs`: エポック数（デフォルト: 20）
- `repeat`: データセットの繰り返し回数（デフォルト: 20）
- `batch_size`: バッチサイズ（デフォルト: 5）
- `save_every_n_epochs`: チェックポイント保存間隔（デフォルト: 10）

**オプティマイザについて:**
- デフォルトでAdafactorを使用（学習率自動調整）
- `lr`パラメータは形式上残していますが、Adafactorの自動調整により無視されます
- AdamWに変更する場合は、コメントアウトされた部分を参照してください

### 4. ControlNet推論 (`cn_eval.py`)

ControlNetを使ったガイダンス付き画像生成を行います。

```python
# cn_eval.py内のパラメータを設定
model_path = "path/to/your/sd15/model"
controlnet_path = "path/to/your/controlnet"
cond_image_path = "path/to/condition/image.png"
```

```bash
python cn_eval.py
```

**マルチControlNet対応:**
```python
controlnet_modules = [
    {"model": controlnet1, "image": cond_image1, "scale": 1.0},
    {"model": controlnet2, "image": cond_image2, "scale": 0.8},
]
```

### 5. ControlNet学習 (`cn_train.py`)

UNetからControlNetを作成し、学習を行います。

```python
# cn_train.py内のパラメータを設定
model_path = "path/to/your/sd15/model"
dataset_path = "dataset"
num_epochs = 5
batch_size = 1
```

```bash
accelerate launch cn_train.py
```

**主要パラメータ:**
- `num_epochs`: エポック数（デフォルト: 5）
- `batch_size`: バッチサイズ（デフォルト: 1）
- `image_size`: 学習画像サイズ（デフォルト: 512）
- `save_every_n_epochs`: チェックポイント保存間隔（デフォルト: 10）

## 📊 データセット形式

### LoRA学習用データセット

```
dataset/
├── image1.png
├── image1.txt
├── image2.png
├── image2.txt
└── ...
```

- 画像ファイル（.png, .jpg等）
- 対応するテキストファイル（同名で拡張子が.txt）
- テキストファイルにはプロンプトを記述

### ControlNet学習用データセット

```
dataset/
├── image1.png
├── image1.txt
├── cond_image1.png
├── image2.png
├── image2.txt
├── cond_image2.png
└── ...
```

- 画像ファイル（ターゲット画像）
- テキストファイル（プロンプト）
- 条件画像ファイル（`cond_`プレフィックス）

## Accelerate設定

初回実行前に、Accelerateの設定を行ってください：

```bash
accelerate config
```

推奨設定:
- Mixed precision: fp16（GPUメモリ節約）
- Gradient accumulation: 必要に応じて設定
- Multi-GPU: 利用可能な場合は有効化

## ユーティリティ

### `utils/utils.py`
- `encode_prompt()`: プロンプトのエンコード
- `prepare_empty_latent()`: 初期潜在変数の準備
- `decode_latents()`: 潜在変数から画像へのデコード
- `get_optimal_torch_dtype()`: 最適なdtype取得

### `utils/lora_utils.py`
- `inject_init_lora_for_unet_textencoder()`: LoRA注入
- `get_model_prefix()`: モデルプレフィックス取得

### `utils/dataset_utils.py`
- `LoRADataset`: LoRA学習用データセットクラス
- `ControlNetDataset`: ControlNet学習用データセットクラス

### `utils/training_manager.py`
- `TrainingManager`: 学習進捗管理とロギング

## 注意事項

1. **モデル形式**: すべてのスクリプトで、`model_path`にはdiffusers形式のモデルパスを指定してください
2. **プロンプト強調**: Stable Diffusion Web UIのようなプロンプト強調記法（例: `(keyword:1.2)`）には対応していません
3. **学習率**: LoRA/ControlNet学習では、Adafactorによる自動学習率調整を使用しているため、`lr`パラメータは実質的に無視されます
4. **GPUメモリ**: ControlNet学習は特にメモリを消費します。必要に応じてバッチサイズを調整してください
5. **LoRA互換性**: LoRAの読み込みは、state_dictのキー形式に依存します。互換性のない形式の場合は、キー変換が必要になる場合があります

## 出力

### 学習時の出力

- **LoRA**: `lora_output/`ディレクトリに`.safetensors`形式で保存
- **ControlNet**: `controlnet_output/`ディレクトリにdiffusers形式で保存
- チェックポイントは`{epoch}_{output_name}`の形式で保存されます

### 推論時の出力

- `generate/`ディレクトリに生成画像が保存されます
- 各サンプリングステップの中間画像も保存されます（`eval.py`）

## 貢献

バグ報告や機能リクエストは、Issuesでお願いします。

## ライセンス

このプロジェクトのライセンスについては、[LICENSE.md](LICENSE.md)を参照してください。

## 謝辞

このプロジェクトは以下のライブラリを使用しています：
- [Diffusers](https://github.com/huggingface/diffusers)
- [Transformers](https://github.com/huggingface/transformers)
- [Accelerate](https://github.com/huggingface/accelerate)
- [PyTorch](https://pytorch.org/)

詳細は[NOTICE](NOTICE)ファイルを参照してください。