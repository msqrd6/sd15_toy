import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL,ControlNetModel
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
#from controlnet_aux import OpenposeDetector, CannyDetector
from PIL import Image
from utils.utils import encode_prompt,prepare_empty_latent,decode_latents,image_to_tensor

# model
model_path = ""

# output
output_dir = "generate"

# parameter
guidance_scale = 7.5
sampling_steps = 20
width, height = 512,512

prompts =  ""
negative_prompt = ""

# dtype, device
device = torch.device("cuda")
dtype = torch.float16

# load models
tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer",local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder",torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)
scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
scheduler.set_timesteps(sampling_steps, device=device) 


# controlnet setting
class ControlNetModule:
    def __init__(
            self,
            model_path,
            image_path,
            guidance_start=0,
            guidance_end=1,
            cond_scale=1,
            pre_processer = None,
            size : tuple[int,int]= None
            ):
        
        self.controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
        self.condition_scales = [cond_scale if guidance_start <= t / sampling_steps <= guidance_end else 0.0 for t in range(sampling_steps)]

        image = Image.open(image_path).convert("RGB")
        if size is not None:
            image = image.resize(size,resample=Image.NEAREST)

        if pre_processer is not None:
            image = pre_processer(image)
        self.condition_image_tensor = image_to_tensor(image)
        self.condition_image_tensor = self.condition_image_tensor.to(dtype=dtype,device=device)

controlnet_modules = [
    ControlNetModule(
        model_path=r"E:\ダウンロード\30_reference_image",
        image_path=r"E:\ダウンロード\C350_B026_005.png",
        guidance_start=0,
        guidance_end=0.8,
        cond_scale=1,
        pre_processer=None,
        size=(512,512)
    )
]

# generate
positive_embeds, negative_embeds = encode_prompt(prompts, tokenizer, text_encoder,negative_prompt=negative_prompt)
prompt_embeds = torch.cat([negative_embeds, positive_embeds])

latents = prepare_empty_latent(width,height,vae)

for i, t in enumerate(tqdm(scheduler.timesteps)):
    with torch.no_grad():
        latent_input = scheduler.scale_model_input(latents, t)
        down_samples_sum = None
        mid_sample_sum = None

        for controlnet in controlnet_modules:
            cond_scale = controlnet.condition_scales[i]
            controlnet_out = controlnet.controlnet(
                latent_input,
                t,
                encoder_hidden_states=positive_embeds,
                controlnet_cond=controlnet.condition_image_tensor,
                conditioning_scale=cond_scale,
                guess_mode=True
            )
            if down_samples_sum is None:
                down_samples_sum = controlnet_out.down_block_res_samples
                mid_sample_sum = controlnet_out.mid_block_res_sample
            else:
                down_samples_sum = [
                    a + b for a, b in zip(down_samples_sum, controlnet_out.down_block_res_samples)
                ]
                mid_sample_sum = mid_sample_sum + controlnet_out.mid_block_res_sample

        # ControlNet 出力を [無条件 (ゼロ), 条件付き] に拡張
        down_block_res_samples = [
            torch.cat([torch.zeros_like(d), d]) for d in down_samples_sum
        ]
        mid_block_res_sample = torch.cat([
            torch.zeros_like(mid_sample_sum), mid_sample_sum
        ])

        noise_pred = unet(
            latent_input.repeat(2, 1, 1, 1),  # バッチを2倍に（[uncond, text]）
            t,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        # CFGで合成
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latent_input).prev_sample

        latent = decode_latents(latents,vae)
        latent.save(f"{output_dir}/gen_{i}.png")





