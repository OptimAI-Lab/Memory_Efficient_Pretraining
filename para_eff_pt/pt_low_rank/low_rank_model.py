import os
import math
import json

from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoConfig


@dataclass
class LoRaFaConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    trainable_scaling: bool = False


class LoRaFaModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        target_modules,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        trainable_scaling=False,
    ):
        if r < 0:
            raise ValueError("r must be nonnegative.")

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_dropout = lora_dropout
        self.lora_alpha = lora_alpha
        self.trainable_scaling = trainable_scaling
        self.target_modules = target_modules
        self.parameterized_modules = []

        self._config = LoRaFaConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            trainable_scaling=trainable_scaling,
        )

        # patch methods
        self.forward = self.wrapped_model.forward

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print(f"Reparameterized module: {module_name}")
            # self.parameterized_modules.append(module_name)
            new_module = LoRaFaLinear(
                module.in_features,
                module.out_features,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                trainable_scaling=self.trainable_scaling,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )

            module.weight = None
            del module

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def save_pretrained(self, path, max_shard_size="100GB"):
        # TODO
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "lorafa_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        # TODO
        with open(os.path.join(path, "lorafa_config.json"), "r") as f:
            lorafa_config = json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)
        if "keep_original" in lorafa_config:
            print("WARNING: keep_original is deprecated. Use lora_only instead.")
            print(f"keep_original: {lorafa_config['keep_original']}")
            lorafa_config["lora_only"] = not lorafa_config.pop("keep_original")
            lorafa_config["keep_original_weights"] = not lorafa_config["lora_only"]

        if "trainable_scaling" not in lorafa_config:
            lorafa_config["trainable_scaling"] = False

        model = cls(base_model, **lorafa_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model


class LoRaFaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: float = 32,
        lora_dropout: float = 0.0,
        trainable_scaling: bool = False,
        bias=True,
        device=None,
        dtype=None,
    ):
        """
        Reparameterized low rank linear layer
                    x W_a @ W_b * lora_alpha / r
        Notice that scale = lora_alpha / r.
        Notice that this class cannot be wrapped to linear layer and thus cannot be used for fine-tune
        For fine-tune, please refer to ... TODO
        """
        super().__init__()
        # nn.Module.__init__(self)
        if r < 0:
            raise ValueError("r must be non-negative.")

        if bias:
            self.bias = Parameter(
                torch.zeros(
                    out_features, device=device, dtype=dtype, requires_grad=True
                )
            )
            a = 1 / math.sqrt(out_features)
            nn.init.uniform_(self.bias, -a, a)
        else:
            self.register_parameter("bias", None)

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.trainable_scaling = trainable_scaling
        self.device = device
        self.dtype = dtype

        self.lora_A = nn.Linear(in_features, r, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_B.weight)

        # self.lora_A = nn.Parameter(
        #     torch.empty(r, in_features, dtype=dtype, device=device), requires_grad=True
        # )
        # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # self.lora_B = nn.Parameter(
        #     torch.empty(out_features, r, dtype=dtype, device=device), requires_grad=True
        # )
        # nn.init.zeros_(self.lora_B)
        
        if trainable_scaling:
            self.scaling = nn.Parameter(
                torch.tensor([1.0], device=device, dtype=dtype), requires_grad=True
            )
        else:
            self.scaling = self.lora_alpha / self.r
            
     

    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()

        return self.scaling

    def forward(self, x: Tensor):
        """
        Input x : [..., in_dim] and Output [..., out_dim]
        """
        out = 0
        if self.r > 0:
            out += (
                self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale()
            )
      
        
    
            
        if self.bias is not None:
            out += self.bias
        return out
