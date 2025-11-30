import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, ControlNetModel
from transformers import CLIPTokenizer, CLIPTextModel, Adafactor
from transformers.optimization import AdafactorSchedule
from accelerate import Accelerator

from utils.utils import get_optimal_torch_dtype, get_trainable_params
from utils.dataset_utils import ControlNetDataset
from utils.trmn import TrainingManager

# model
model_path = "" #diffusers形式

#dataset
dataset_path = "dataset"

# output
output_dir = "controlnet_output"
output_name = "controlnet"

# train prameter
#lr = 1e-5
repeat = 1
batch_size = 1
num_epochs = 40
gradient_accumulation_steps = 1
save_every_n_epochs = 10
image_size = 512

# accelerator, dtype, device
accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps,
)
device = accelerator.device
dtype, train_model_dtype = get_optimal_torch_dtype(accelerator.mixed_precision)

# load model
tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer",local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder",torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)
scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

# create controlnet
controlnet = ControlNetModel.from_unet(unet).to(device=device,dtype=train_model_dtype)

# train settings
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)

# prepare dataset
dataset = ControlNetDataset(dataset_path, vae, tokenizer, text_encoder, image_size,repeat=repeat)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


optimizer = Adafactor(
    get_trainable_params(controlnet),
    scale_parameter=True,
    relative_step=True,
    warmup_init=True,
    lr=None,  # 自動調整
)

lr_scheduler = AdafactorSchedule(
    optimizer,
    )

# prepare (acceleratorに渡して wrap する)
controlnet, unet, text_encoder, optimizer, lr_scheduler, dataloader = accelerator.prepare(
    controlnet, unet, text_encoder, optimizer, lr_scheduler,  dataloader
)

def _save_weight(output_name):
    save_dir = os.path.join(output_dir,output_name)
    os.makedirs(save_dir,exist_ok=True)

    controlnet.save_pretrained(
        save_directory=save_dir,
        safe_serialization=True  # safetensors形式で保存
        )

tm = TrainingManager(training_models=[controlnet],
                     dataloader=dataloader,
                     num_epochs=num_epochs,
                     save_every_n_epochs=save_every_n_epochs,
                     log_interval=100,
                     )

# train
tm.train_mode()
for epoch in tm.epochs:
    for image_latents, positive_embeds, cond_tensors in dataloader:
        noise = torch.randn_like(image_latents)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (image_latents.size(0),), device=device).long()

        # ノイズ付加
        noisy_latents = scheduler.add_noise(image_latents, noise, t)
        
        with accelerator.autocast():
            down_block_res_samples,mid_block_res_sample = controlnet(
                noisy_latents,
                t,
                encoder_hidden_states=positive_embeds,
                controlnet_cond=cond_tensors,
                conditioning_scale=1.0,
                return_dict=False,
            )
                
            noise_pred = unet(
                noisy_latents, 
                t, 
                encoder_hidden_states=positive_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                ).sample
            
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="none")
        loss = loss.mean()

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        tm.batch_step(loss.item())

    if tm.is_savepoint():
        _save_weight(f"{tm.current_epoch}_{output_name}")

    tm.plot(f"log_{tm.current_epoch}",f"{output_dir}/plot")
    tm.epoch_step()