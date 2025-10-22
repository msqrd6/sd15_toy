import os
import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, Adafactor
from transformers.optimization import AdafactorSchedule
from safetensors.torch import save_file
from accelerate import Accelerator
from itertools import chain

from utils.dataset_utils import LoRADataset
from utils.utils import get_optimal_torch_dtype
from utils.lora_utils import inject_init_lora_for_unet_textencoder, get_model_prefix
from utils.training_manager import TrainingManager

# base_model
model_path = ""

# train_data
dataset_path = "dataset"

# output
output_dir = "lora_output"
output_name = "test"

# training parameter
rank = 128
alpha = 64
image_size = 512

num_epochs = 20
repeat = 20
batch_size = 5
save_every_n_epochs = 10
lr = 1e-3 # Adafactorで自動調整


# accelerator, dtype, device
accelerator = Accelerator()
device = accelerator.device
dtype, train_model_dtype = get_optimal_torch_dtype(accelerator.mixed_precision) # dtype = load and default, train_dtype = use train model


# load models
tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer")
text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(f"{model_path}/vae", torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(f"{model_path}/unet", torch_dtype=dtype).to(device)
scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

# 元パラメータの凍結
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)

# lora注入前にデータセットを作る
dataset = LoRADataset(dataset_path, vae, tokenizer, text_encoder, image_size,repeat=repeat)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# loraを注入
unet_alphas = inject_init_lora_for_unet_textencoder(unet,rank,alpha,dtype=train_model_dtype)
te_alphas = inject_init_lora_for_unet_textencoder(text_encoder,rank,alpha,dtype=train_model_dtype)
network_alphas = {**unet_alphas,**te_alphas}

trainable_params = list(chain(unet.parameters(),text_encoder.parameters()))

optimizer = Adafactor(
    trainable_params,
    scale_parameter=True,
    relative_step=True,
    warmup_init=True,
    lr=None,  # 自動調整
)

lr_scheduler = AdafactorSchedule(
    optimizer,
    initial_lr=lr
    )

# prepare (acceleratorに渡して wrap する)
unet, text_encoder, optimizer, lr_scheduler, dataloader = accelerator.prepare(
    unet, text_encoder, optimizer, lr_scheduler, dataloader
)

progress = TrainingManager(dataloader,num_epochs,save_every_n_epochs)

def transform_lora_key(key: str) -> str:
    key = key.replace('.lora_A.0.weight', '.lora_A.weight')
    key = key.replace('.lora_B.0.weight', '.lora_B.weight')
    return key

def get_trainable_dict(unet,text_encoder):
    trainable_dict={}
    for model in [unet,text_encoder]:
        prefix = get_model_prefix(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_dict[prefix+"."+name] = param

    return trainable_dict

def _save(output_name,unet,text_encoder,network_alphas):
    os.makedirs(output_dir,exist_ok=True)
    unet_to_save = accelerator.unwrap_model(unet)
    text_encoder_to_save = accelerator.unwrap_model(text_encoder)
    trained_dict = {
        transform_lora_key(name): param.detach()
        for name, param in get_trainable_dict(unet_to_save,text_encoder_to_save).items()
    }
    # alpha 値も追加
    lora_state_dict = {**trained_dict, **network_alphas}
    # LoRAの重み保存
    save_file(lora_state_dict, os.path.join(output_dir, output_name+".safetensors"))


# 学習ループ
unet.train()
text_encoder.train()
for epoch in progress.epochs:
    for latents, positive_embeds in progress.dataloader:
        noise = torch.randn_like(latents)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            
        noisy_latents = scheduler.add_noise(latents, noise, t)

        with accelerator.autocast():
            noise_pred = unet(
                noisy_latents,
                t, 
                encoder_hidden_states=positive_embeds
                ).sample

        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="none")
        loss = loss.mean()

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress.step(loss.item())
    
    if progress.is_checkpoint():
        _save(f"{progress.current_epoch}_{output_name}",unet,text_encoder,network_alphas)
    
    progress.epoch_step()