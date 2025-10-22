import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
import torch
import torch.nn as nn
from typing import Union
from copy import deepcopy

def get_optimal_torch_dtype(dtype_name:str):
    if dtype_name == "fp16":
        return torch.float16,torch.float32
    elif dtype_name == "bf16":
        return torch.bfloat16,torch.bfloat16
    else:
        return torch.float32,torch.float32

def image_to_tensor(image: Image.Image)->torch.Tensor:
    width, height = image.size

    transform = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor() # これにより [0, 1] になる
    ])

    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def load_image(img_path:str, size:tuple[int,int]=None, resample=Image.NEAREST, round_size = True) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    
    if round_size:
        width, height = img.size if size is None else size
        width = max(8, int(round(width / 8) * 8))
        height = max(8, int(round(height / 8) * 8))
        size = (width,height)

    img = img.resize(size, resample=resample)
    return image_to_tensor(img)


def encode_image(image_tensors: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    image_tensors = (image_tensors * 2 - 1).to(dtype=vae.dtype,device=vae.device) # [0, 1] -> [-1, 1] 
    with torch.no_grad():
        encoded = vae.encode(image_tensors)
        latents = encoded.latent_dist.sample() * vae.config.scaling_factor
    return latents # shape: (1, 4, 64, 64)


def decode_latents(latents: torch.Tensor, vae: AutoencoderKL, return_tensor:bool = False) -> Union[Image.Image,list[Image.Image],torch.Tensor]:
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image_tensors = vae.decode(latents.to(dtype=vae.dtype)).sample

    image_tensors = (image_tensors / 2 + 0.5).clamp(0, 1) # [-1, 1] -> [0, 1]
    if return_tensor:
        return image_tensors
    
    image_np = image_tensors.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    image_uint8 = (image_np * 255).round().astype("uint8")

    if image_tensors.shape[0] == 1:
        return Image.fromarray(image_uint8[0])

    pil_images = [Image.fromarray(img) for img in image_uint8]
    return pil_images
    

def encode_prompt(
        prompt, 
        tokenizer, 
        text_encoder, 
        negative_prompt=None
        )-> tuple[torch.Tensor,torch.Tensor]:
    
    def get_embeds(prompts):
        inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.to(device=text_encoder.device)

        with torch.no_grad():
            embeds = text_encoder(input_ids)[0]  # (batch, seq_len, dim)
        return embeds
    
    #バッチ対応用
    prompt = [prompt] if isinstance(prompt,str) else prompt
    negative_prompt = [""] * len(prompt) if negative_prompt is None else negative_prompt

    positive_embeds = get_embeds(prompt)
    negative_embeds = get_embeds(negative_prompt)

    return positive_embeds, negative_embeds


def prepare_empty_latent(
        width,
        height,
        vae,
        batch_size = 1,
        scheduler=None
        ):
    latents = torch.randn((batch_size, 4, height//8, width//8), device=vae.device, dtype=vae.dtype)
    init_noise_sigm = scheduler.init_noise_sigma if scheduler is not None else 1.0
    return latents * init_noise_sigm


    
def show_model_param_status(model_or_dict,name_only=False):
    if isinstance(model_or_dict,nn.Module): 
        for name, param in model_or_dict.named_parameters():
            if name_only:
                print(name)
            else:
                print("")
                print(name)
                print(f"{param.dtype}, {param.device}, {param.requires_grad}")
                print(f"{list(param.shape)}")
        l = len(model_or_dict.state_dict())

    if isinstance(model_or_dict,dict): 
        for k, v in model_or_dict.items():
            if name_only:
                print(k)
            else:
                print(f"{v.dtype}, {v.device}, {v.requires_grad} : {k}")
                print(v.shape)
        
        l = len(model_or_dict)
    print("dtype, device, requires_grad : name , shape")
    print(f"length: {l}")


def show_model_module_status(model:nn.Module):
    for name,module in model.named_modules():
        print(name)
        print(type(module))