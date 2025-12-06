import os
import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, Adafactor
from transformers.optimization import AdafactorSchedule
from safetensors.torch import save_file
from accelerate import Accelerator

from utils.dataset_utils import LoRADataset
from utils.utils import get_optimal_torch_dtype, get_trainable_params
from utils.lora_utils import inject_init_lora_into_model, get_lora_dict_from_model
from utils.trmn import TrainingManager

# base_model
model_path = "diffusers_models\example_diffusers_model"

# train_data
dataset_path = "dataset"

# output
output_dir = "lora_output"
output_name = "lora"

# lora parameter
rank = 64
alpha = 32
dropout = 0.0

# train prameter
#lr = 1e-5
repeat = 1
batch_size = 1
num_epochs = 20
gradient_accumulation_steps = 1
save_every_n_epochs = 10
image_size = 512


# accelerator, dtype, device
accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps
)
device = accelerator.device
dtype, train_model_dtype = get_optimal_torch_dtype(accelerator.mixed_precision) # dtype = load and default, train_dtype = use train model


# load models
tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer")
text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(f"{model_path}/vae", torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(f"{model_path}/unet", torch_dtype=dtype).to(device)
scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

# freeze parametors
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)

# dataset
dataset = LoRADataset(dataset_path, vae, tokenizer, text_encoder, image_size,repeat=repeat)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# inject lora
inject_init_lora_into_model(unet,
                           rank,
                           alpha,
                           dropout,
                           inject_layer_key=["attentions"],
                           linear=True,
                           conv2d=False,
                           )


optimizer = Adafactor(
    get_trainable_params(unet),
    scale_parameter=True,
    relative_step=True,
    warmup_init=True,
    lr=None,  # 自動調整
)

lr_scheduler = AdafactorSchedule(
    optimizer,
    )

# prepare (acceleratorに渡して wrap する)
unet, text_encoder, optimizer, lr_scheduler, dataloader = accelerator.prepare(
    unet, text_encoder, optimizer, lr_scheduler, dataloader
)

tm = TrainingManager(trainable_modules=[unet],
                     frozen_modules=[text_encoder,vae],
                     dataloader=dataloader,
                     num_epochs=num_epochs,
                     save_every_n_epochs=save_every_n_epochs,
                     log_interval=100,
                     )


def _save(output_name,unet):
    os.makedirs(output_dir,exist_ok=True)
    unet_to_save = accelerator.unwrap_model(unet)
    lora_state_dict = get_lora_dict_from_model(unet_to_save)
    save_file(lora_state_dict, os.path.join(output_dir, output_name+".safetensors"))


# 学習ループ
tm.train_mode()
for epoch in tm.epochs:
    for latents, positive_embeds in tm.dataloader:
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

        tm.batch_step(loss.item())
    
    if tm.is_savepoint():
        _save(f"{tm.current_epoch}_{output_name}",unet)
    
    tm.plot( f"log_{tm.current_epoch}",f"{output_dir}\plot")
    tm.epoch_step()