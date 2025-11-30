# SD15_Toy

Stable Diffusion 1.5の学習・推論を簡潔に行うためのPyTorchベースのツールキットです。LoRAとControlNetの学習・推論に対応しています。

## 特徴

- **シンプルな実装**: 最小限の依存関係で、SD1.5の学習・推論を実現
- **LoRA対応**: UNetとTextEncoderへのLoRA注入と学習
- **ControlNet対応**: マルチControlNetによるガイダンス付き生成
- **Accelerate統合**: 分散学習とmixed precisionに対応
- **柔軟なデータセット**: カスタムデータセットで簡単に学習可能
- **TrainingManager**: 学習進捗の自動管理とロギング機能

## 必要要件

- Python 3.8+
- CUDA対応GPU
- PyTorch 2.0+
- Diffusers
- Transformers
- Accelerate
- safetensors

## インストール

```bash
git clone https://github.com/msqrd6/sd15_toy.git
cd sd15_toy

# 1. PyTorchのインストール（CUDA対応版）
# CUDA 11.8の場合:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1の場合:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU版の場合:
# pip install torch torchvision

# 2. その他の依存関係のインストール
pip install -r requirements.txt
```

> **注意**: 使用しているCUDAのバージョンに合わせてPyTorchをインストールしてください。詳細は[PyTorch公式サイト](https://pytorch.org/get-started/locally/)を参照してください。

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
    ├── trmn.py          # TrainingManager（学習進捗管理）
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
strength = 0.8  # LoRAの強度
```

```bash
python lora_eval.py
```

**Kohya形式のLoRA変換:**
```python
# Kohya形式のLoRAを使用する場合は、以下のコードのコメントを外す
weights, te_lora_dict = convert_injectable_dict_from_khoya_weight(weights)
inject_pretrained_lora_into_model(text_encoder, te_lora_dict, strength)
```

### 3. LoRA学習 (`lora_train.py`)

UNetにLoRAを注入して学習を行います。

```python
# lora_train.py内のパラメータを設定
model_path = "path/to/your/sd15/model"
dataset_path = "dataset"
rank = 64
alpha = 32
num_epochs = 20
batch_size = 1
```

```bash
accelerate launch lora_train.py
```

**主要パラメータ:**
- `rank`: LoRAのランク（デフォルト: 64）
- `alpha`: LoRAのアルファ値（デフォルト: 32）
- `dropout`: ドロップアウト率（デフォルト: 0.0）
- `num_epochs`: エポック数（デフォルト: 20）
- `repeat`: データセットの繰り返し回数（デフォルト: 1）
- `batch_size`: バッチサイズ（デフォルト: 1）
- `save_every_n_epochs`: チェックポイント保存間隔（デフォルト: 10）
- `log_interval`: ロギング間隔（デフォルト: 50）

**オプティマイザについて:**
- デフォルトでAdafactorを使用（学習率自動調整）
- `lr`パラメータは`None`に設定（Adafactorの自動調整を使用）

**LoRA注入設定:**
```python
inject_init_lora_into_model(
    unet,
    rank,
    alpha,
    dropout,
    inject_layer_key=["attentions"],  # 注入する層のキーワード
    linear=True,   # Linear層に注入
    conv2d=False,  # Conv2d層には注入しない
)
```

### 4. ControlNet推論 (`cn_eval.py`)

ControlNetを使ったガイダンス付き画像生成を行います。

```python
# cn_eval.py内のパラメータを設定
model_path = "path/to/your/sd15/model"
prompts = "your prompt"
```

**ControlNetModule設定:**
```python
controlnet_modules = [
    ControlNetModule(
        model_path="path/to/controlnet",
        image_path="path/to/condition/image.png",
        guidance_start=0,      # ガイダンス開始位置（0-1）
        guidance_end=0.8,      # ガイダンス終了位置（0-1）
        cond_scale=1,          # ガイダンス強度
        pre_processer=None,    # 前処理関数（オプション）
        size=(512, 512)        # リサイズサイズ
    )
]
```

```bash
python cn_eval.py
```

**マルチControlNet対応:**
複数のControlNetModuleをリストに追加することで、マルチControlNetに対応します。

### 5. ControlNet学習 (`cn_train.py`)

UNetからControlNetを作成し、学習を行います。

```python
# cn_train.py内のパラメータを設定
model_path = "path/to/your/sd15/model"
dataset_path = "dataset"
num_epochs = 40
batch_size = 1
save_every_n_epochs = 10
```

```bash
accelerate launch cn_train.py
```

**主要パラメータ:**
- `num_epochs`: エポック数（デフォルト: 40）
- `batch_size`: バッチサイズ（デフォルト: 1）
- `image_size`: 学習画像サイズ（デフォルト: 512）
- `save_every_n_epochs`: チェックポイント保存間隔（デフォルト: 10）
- `repeat`: データセットの繰り返し回数（デフォルト: 1）
- `log_interval`: ロギング間隔（デフォルト: 100）

## データセット形式

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

## TrainingManager

`TrainingManager`は学習進捗の管理とロギングを自動化するユーティリティクラスです。

**主な機能:**
- 自動的なエポック・バッチ進捗管理
- 損失のロギングとプロット
- チェックポイント保存タイミングの管理
- バリデーションループのサポート

**使用例:**
```python
tm = TrainingManager(
    training_models=[unet],
    dataloader=dataloader,
    num_epochs=num_epochs,
    save_every_n_epochs=save_every_n_epochs,
    log_interval=50,  # 50バッチごとにログ
)

tm.train_mode()
for epoch in tm.epochs:
    for batch_data in tm.dataloader:
        # 学習処理
        loss = compute_loss(batch_data)
        
        # 損失を記録
        tm.batch_step(loss.item())
    
    # チェックポイント保存
    if tm.is_savepoint():
        save_model(f"{tm.current_epoch}_model")
    
    tm.epoch_step()

# 学習曲線をプロット
tm.plot(name="training_loss", output_dir="output")
```

**バリデーション機能:**
```python
tm = TrainingManager(
    training_models=[unet],
    dataloader=train_dataloader,
    num_epochs=num_epochs,
    valid_dataloader=valid_dataloader,
    valid_every_n_epochs=5,
    n_batches_valid=10,  # バリデーションバッチ数
)

for epoch in tm.epochs:
    # 学習ループ
    for batch_data in tm.dataloader:
        # ...
        tm.batch_step(loss.item())
    
    # バリデーション
    if tm.is_validpoint():
        tm.valid_start()
        for valid_batch in tm.valid_dataloader:
            valid_loss = compute_loss(valid_batch)
            tm.valid_step(valid_loss)
        tm.valid_end()
    
    tm.epoch_step()
```

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
- `get_trainable_params()`: 学習可能なパラメータの取得
- `image_to_tensor()`: 画像をテンソルに変換

### `utils/lora_utils.py`
- `inject_init_lora_into_model()`: 初期化されたLoRAを注入
- `inject_pretrained_lora_into_model()`: 学習済みLoRAを注入
- `get_lora_dict_from_model()`: モデルからLoRA state dictを取得
- `remove_lora_from_model()`: モデルからLoRAを削除
- `marge_lora_and_weight()`: LoRAとベースウェイトをマージ

### `utils/dataset_utils.py`
- `LoRADataset`: LoRA学習用データセットクラス
- `ControlNetDataset`: ControlNet学習用データセットクラス

### `utils/trmn.py`
- `TrainingManager`: 学習進捗管理とロギング
  - `train_mode()` / `eval_mode()`: モード切り替え
  - `batch_step()`: バッチごとの処理
  - `epoch_step()`: エポックごとの処理
  - `is_savepoint()`: チェックポイント保存タイミング判定
  - `is_validpoint()`: バリデーション実行タイミング判定
  - `plot()`: 学習曲線のプロット

### `utils/convert_utils.py`
- `convert_injectable_dict_from_khoya_weight()`: Kohya形式のLoRAウェイトを変換

## 注意事項

1. **モデル形式**: すべてのスクリプトで、`model_path`にはdiffusers形式のモデルパスを指定してください
2. **プロンプト強調**: Stable Diffusion Web UIのようなプロンプト強調記法（例: `(keyword:1.2)`）には対応していません
3. **学習率**: LoRA/ControlNet学習では、Adafactorによる自動学習率調整を使用しているため、`lr`パラメータは`None`に設定してください
4. **GPUメモリ**: ControlNet学習は特にメモリを消費します。必要に応じてバッチサイズを調整してください
5. **LoRA互換性**: LoRAの読み込みは、state_dictのキー形式に依存します。Kohya形式の場合は`convert_utils`を使用して変換してください

## 出力

### 学習時の出力

- **LoRA**: `lora_output/`ディレクトリに`.safetensors`形式で保存
- **ControlNet**: `controlnet_output/`ディレクトリにdiffusers形式で保存
- チェックポイントは`{epoch}_{output_name}`の形式で保存されます
- 学習曲線は`plot()`メソッドで保存可能

### 推論時の出力

- `generate/`ディレクトリに生成画像が保存されます
- 各サンプリングステップの中間画像も保存されます

## ライセンス

このプロジェクトのライセンスについては、[LICENSE.md](LICENSE.md)を参照してください。

## 謝辞

このプロジェクトは以下のライブラリを使用しています：
- [Diffusers](https://github.com/huggingface/diffusers)
- [Transformers](https://github.com/huggingface/transformers)
- [Accelerate](https://github.com/huggingface/accelerate)
- [PyTorch](https://pytorch.org/)

詳細は[NOTICE](NOTICE)ファイルを参照してください。
