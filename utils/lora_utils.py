import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import torch.nn as nn
import math
from typing import Union
from copy import deepcopy


def get_model_prefix(model):
    if isinstance(model,UNet2DConditionModel):
        return "unet"
    elif isinstance(model,CLIPTextModel):
        return "text_encoder"
    else:
        return None

def get_module_by_key(model, key):
    parts = key.split('.')
    module = model
    for p in parts[:-1]:
        if p.isdigit():
            module = module[int(p)]
        else:
            module = getattr(module, p)
    return module, parts[-1]

def inject_empty_lora_layer(model,module_name):
    parent_module = model
    path = module_name.split(".")
    for p in path[:-1]:
        parent_module = getattr(parent_module,p)
    last_name = path[-1]
    base_layer = getattr(parent_module, last_name)
    lora_layer = LoRA(base_layer)
    setattr(parent_module, last_name, lora_layer)
    return lora_layer

def inject_init_lora_for_unet_textencoder(model, rank=4, alpha=1.0, dropout=0.0):
    network_alphas = {}
    #loraを注入する層か判定(unetはattention,teはmlpかattnに注入)
    def needs_lora_injection(module_name):
        if isinstance(model,UNet2DConditionModel):
            if "attention"  in module_name:
                return True
        elif isinstance(model,CLIPTextModel):
            if "mlp" in module_name or "self_attn" in module_name:
                return True
        return False

    for module_name, module in model.named_modules():
        if not needs_lora_injection(module_name):
            continue
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prefix = get_model_prefix(model)
            network_alphas[prefix+"."+module_name+".alpha"] = torch.tensor(alpha)
            lora_layer = inject_empty_lora_layer(model,module_name)
            lora_layer.init_lora(rank,alpha,dropout)
    # これ結構重要
    model = model.to(device=model.device,dtype=model.dtype)

    return network_alphas

def marge_lora_and_weight(lora_state_dict,base_state_dict,strength=1.0):
    output_state_dict = deepcopy(base_state_dict)
    for key, value in lora_state_dict.items():
        if not "lora_A" in key: continue
        base_key = key.split(".lora_A.")[0]
        lora_A = value
        lora_B = lora_state_dict.get(base_key + '.lora_B.weight')
        rank = lora_A.shape[0]
        # .alphaが存在しない場合はrank/2を代入
        alpha = lora_state_dict.get(base_key + '.alpha', rank/2)
        scale = strength*alpha/rank
        if lora_A.dim() == 4:
            delta_W = (lora_B.squeeze() @ lora_A.squeeze())
            delta_W = delta_W.unsqueeze(-1).unsqueeze(-1)
        else:
            delta_W = (lora_B @ lora_A)

        with torch.no_grad():
            output_state_dict[base_key+".weight"] += scale*delta_W
    return output_state_dict

def separate_lora_from_model(model:nn.Module,out_model_state_dict=False):
    lora_state_dict = {}
    model_state_dict = {}

    with torch.no_grad():
        for key, value in model.state_dict().items():
            if "lora" in key:
                out_key = key.replace("0.weight","weight")
                lora_state_dict[out_key] = value
            elif out_model_state_dict:
                if "base_layer.weight" in key:
                    out_key = key.replace("base_layer.weight","weight")
                elif "base_layer.bias" in key:
                    out_key = key.replace("base_layer.bias","bias")
                else:
                    out_key = key
                model_state_dict[out_key] = value

    if not out_model_state_dict:
        return lora_state_dict
    
    return lora_state_dict, model_state_dict


def inject_pretrained_lora_for_model(base_model,lora_state_dict,strength=1.0):
    for key, value in lora_state_dict.items():
        if not "lora_A" in key: continue
        base_key = key.split(".lora_A.")[0]
        lora_A = value
        lora_B = lora_state_dict.get(base_key + '.lora_B.weight')
        rank = lora_A.shape[0]
        # .alphaが存在しない場合はrank/2を代入
        alpha = lora_state_dict.get(base_key + '.alpha', rank/2)
        
        lora_layer = inject_empty_lora_layer(base_model,base_key)
        lora_layer.load_weight(lora_A,lora_B,strength,alpha)
    base_model = base_model.to(device=base_model.device,dtype=base_model.dtype)


class LoRA(nn.Module):
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        self.scales = []
        self.dropouts = nn.ModuleList() # Dropout層を保持
        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def init_lora(self,rank,alpha,dropout=0.0):
        self.scales.append(alpha / rank if rank > 0 else 1.0)
        self.dropouts.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())

        if isinstance(self.base_layer, nn.Linear):
            self.lora_A.append(nn.Linear(self.base_layer.in_features, rank, bias=False))
            self.lora_B.append(nn.Linear(rank, self.base_layer.out_features, bias=False))
        elif isinstance(self.base_layer, nn.Conv2d):
            self.lora_A.append(nn.Conv2d(self.base_layer.in_channels, rank, kernel_size=1, bias=False))
            self.lora_B.append(
                nn.Conv2d(rank, self.base_layer.out_channels, kernel_size=1, stride=self.base_layer.stride, padding=self.base_layer.padding, bias=False)
            )

        nn.init.kaiming_uniform_(self.lora_A[-1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[-1].weight)


    def load_weight(self, lora_A, lora_B, strength=1.0, alpha=1.0, dropout=0.0, idx=None):
        idx = -1 if idx is None else idx
        rank = lora_A.shape[0]
        alpha = alpha*strength
        self.init_lora(rank,alpha,dropout)
        self.lora_A[idx].weight.data.copy_(lora_A)
        self.lora_B[idx].weight.data.copy_(lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.base_layer(x)

        for i in range(len(self.lora_A)):
            a_module, b_module = self.lora_A[i], self.lora_B[i]
            scale = self.scales[i]
            dropout = self.dropouts[i]
            w += scale * dropout(b_module(a_module(x)))
        
        return w
    

class SingLoRA(nn.Module):
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        self.scales = []
        self.dropouts = nn.ModuleList() # Dropout層を保持
        self.lora_A = nn.ParameterList()
        self.u_t = 1.0

        self.features = max(self.base_layer.in_features,self.base_layer.out_features)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def init_lora(self,rank,alpha,dropout=0.0):
        self.scales.append(alpha / rank if rank > 0 else 1.0)
        self.dropouts.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())
        self.lora_A.append(nn.Parameter(torch.zeros([self.features,rank])))
        nn.init.kaiming_uniform_(self.lora_A[-1], a=math.sqrt(5))

    def set_u_t(self,u_t):
        self.u_t = u_t

    def add_lora(self, lora_A, alpha=1.0, dropout=0.0, idx = None):
        idx = -1 if idx is None else idx
        rank = lora_A.shape[0]
        self.init_lora(rank,alpha,dropout)
        self.lora_A[idx].data.copy_(lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.base_layer(x)
        for i in range(len(self.lora_A)):
            ab = x @ self.lora_A[i][:self.base_layer.in_features,:] @ self.lora_A[i][:self.base_layer.out_features,:].T
            w += self.u_t * self.scales[i] * ab
        return w


def inject_initial_sing_lora(model, rank=4, alpha=1.0, dropout=0.0,dtype=torch.float32):
    network_alphas = {}
    #loraを注入する層か判定(unetはattention,teはmlpかattnに注入)
    def needs_lora_injection(module_name):
        if isinstance(model,UNet2DConditionModel):
            if "attention"  in module_name:
                return True
        elif isinstance(model,CLIPTextModel):
            if "mlp" in module_name or "self_attn" in module_name:
                return True
        return False

    for module_name, module in model.named_modules():
        if not needs_lora_injection(module_name):
            continue

        if isinstance(module, nn.Linear):# or isinstance(module, nn.Conv2d):
            prefix = get_model_prefix(model)
            network_alphas[prefix+"."+module_name+".alpha"] = torch.tensor(alpha)
            parent_module = model
            path = module_name.split(".")
            # 親モジュールを取得
            for p in path[:-1]:
                parent_module = getattr(parent_module, p)
            last_name = path[-1]

            base_layer = getattr(parent_module, last_name)
    
            lora_layer = SingLoRA(base_layer)
            lora_layer.init_lora(rank,alpha,dropout)

            # モジュールを置換
            setattr(parent_module, last_name, lora_layer)
    # これ結構重要
    model = model.to(device=model.device)

    return network_alphas

def u_t_manager(model,current_step,total_step,warmup=0.1):
    warmup_step = int(warmup*total_step)
    if current_step > warmup_step:
        return 
    u_t = current_step/warmup_step
    singlora_set_u_t(model,u_t)

    
def singlora_set_u_t(model:nn.Module,u_t):
    for module in model.modules():
        if isinstance(module,SingLoRA):
            module.set_u_t(u_t)
