import torch
from tqdm import tqdm
from utils.utils import encode_prompt,encode_image,load_image
import os

# データセットの準備（例：画像とキャプション）
class LoRADataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root_dir, 
                 vae, 
                 tokenizer, 
                 text_encoder, 
                 size=512, 
                 repeat=1,
                 default_caption = "",
                 cache_on_cpu = False,
                 image_dir = "images",
                 caption_dir = "captions",
                 
                 ):
        self.vae = vae
        self.image_dir = os.path.join(root_dir, image_dir)
        self.caption_dir = None if caption_dir is None else os.path.join(root_dir, caption_dir)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.size = (size,size)
        self.repeat=repeat
        self.cache_on_cpu = cache_on_cpu

        self.image_filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.latents = {}
        self.positive_embeds = {}
        # latent 変換
        for img_filename in tqdm(self.image_filenames):
            #image
            img_path = os.path.join(self.image_dir, img_filename)
            image = load_image(img_path,self.size)
            latent = encode_image(image,vae)
            self.latents[img_filename] = latent.squeeze(0).cpu() if self.cache_on_cpu else latent.squeeze(0)
            #text_embeds
            if self.caption_dir is None:
                caption = default_caption
            else:
                txt_path = os.path.join(self.caption_dir, os.path.splitext(img_filename)[0] + ".txt")
                with open(txt_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
            # enbedsを取得

            positive_embeds, _ = encode_prompt(caption,self.tokenizer,self.text_encoder)
            self.positive_embeds[img_filename] = positive_embeds.squeeze(0).cpu() if self.cache_on_cpu else positive_embeds.squeeze(0)

    def __len__(self):
        return len(self.image_filenames) * self.repeat

    def __getitem__(self, idx):
        true_idx = idx % len(self.image_filenames)
        img_filename = self.image_filenames[true_idx]

        latent = self.latents[img_filename]
        positive_embeds = self.positive_embeds[img_filename]

        if self.cache_on_cpu:
            device = self.vae.device
            latent = latent.to(device=device)
            positive_embeds = positive_embeds.to(device=device)

        return latent, positive_embeds
    
    

class ControlNetDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root_dir, 
                 vae, 
                 tokenizer, 
                 text_encoder, 
                 size=512, 
                 repeat=1, 
                 scale=1.0, 
                 default_caption="",
                 cache_on_cpu=False,
                 image_dir = "images",
                 cond_dir = "conditionings",
                 caption_dir = "captions"
                 ):
        
        self.image_dir = os.path.join(root_dir, image_dir)
        self.cond_dir = os.path.join(root_dir, cond_dir)
        self.caption_dir = os.path.join(root_dir, caption_dir)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.size = (size,size)
        self.repeat=repeat
        self.scale = scale
        self.cache_on_cpu = cache_on_cpu

        self.image_filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
 

        self.image_filenames = self.image_filenames[:int(len(self.image_filenames)*self.scale)]
     
        self.image_latents = {}
        self.cond_tensors = {}
        self.positive_embeds = {}
        # latent 変換
        for img_filename in tqdm(self.image_filenames):
            #image
            img_path = os.path.join(self.image_dir, img_filename)
            image = load_image(img_path,self.size)
            image_latent = encode_image(image,vae)
            self.image_latents[img_filename] = image_latent.squeeze(0).cpu() if self.cache_on_cpu else image_latent.squeeze(0)
            #cond
            cond_img_path = os.path.join(self.cond_dir, img_filename)
            cond_img_tensor = load_image(cond_img_path,self.size)
            cond_img_tensor = cond_img_tensor.squeeze(0).to(device=image_latent.device,dtype=image_latent.dtype)
            self.cond_tensors[img_filename] = cond_img_tensor.cpu() if self.cache_on_cpu else cond_img_tensor
            #text_embeds
            txt_path = os.path.join(self.caption_dir, os.path.splitext(img_filename)[0] + ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
            else:
                caption = default_caption
            positive_embeds, _ = encode_prompt(caption,self.tokenizer,self.text_encoder)
            self.positive_embeds[img_filename] = positive_embeds.squeeze(0).cpu() if self.cache_on_cpu else positive_embeds.squeeze(0) 

    def __len__(self):
        return len(self.image_filenames) * self.repeat

    def __getitem__(self, idx):
        true_idx = idx % len(self.image_filenames)
        img_filename = self.image_filenames[true_idx]

        image_latent = self.image_latents[img_filename]
        cond_tensor = self.cond_tensors[img_filename]
        positive_embeds = self.positive_embeds[img_filename]

        if self.cache_on_cpu:
            device = self.vae.device
            image_latent = image_latent.to(device)
            cond_tensor = cond_tensor.to(device)
            positive_embeds = positive_embeds.to(device)

        
        return image_latent, positive_embeds, cond_tensor