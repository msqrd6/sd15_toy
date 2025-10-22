import os
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from utils.utils import encode_prompt, prepare_empty_latent,decode_latents

# base_model
model_path = ""

# output
output_dir = "generate"
os.makedirs(output_dir,exist_ok=True)

# parameter
sampling_steps=20 
guidance_scale = 7.5
width,height= 512,512

prompts =  "sitting a dog"
negative_prompt = ""

# dtype, device
device = torch.device("cuda")
dtype = torch.float16

# load models
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer",local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
scheduler.set_timesteps(sampling_steps,device=device)

# prepare
positive_embeds, negative_embeds = encode_prompt(prompts, tokenizer, text_encoder,negative_prompt=negative_prompt)
prompt_embeds = torch.cat([negative_embeds, positive_embeds])
latents = prepare_empty_latent(width,height,vae,scheduler=scheduler)

# generate
for i, t in enumerate(tqdm(scheduler.timesteps)):
    with torch.no_grad():
        latent_input = torch.cat([latents]*2)
        latent_input = scheduler.scale_model_input(latent_input, t)

        noise_pred = unet(
            latent_input,
            t, 
            encoder_hidden_states=prompt_embeds
            ).sample

        # cfg
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # xt -> xt-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        latent = decode_latents(latents,vae)
        latent.save(f"{output_dir}/gen_{i}.png")




