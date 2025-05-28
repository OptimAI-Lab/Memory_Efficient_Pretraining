import os
import json
import math

from typing import List
from dataclasses import dataclass

import torch
from torch import nn
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig


@dataclass
class SpLoRaConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    sp_ratio: float
    sp_type: str
    target_modules: List[str]
    trainable_scaling: bool = False
    random_subspace: bool = False


class SpLoRaModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        target_modules,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        sp_ratio=0.01,
        sp_type="random",
        trainable_scaling=False,
        random_subspace=False,
    ):
        if r < 0:
            raise ValueError("r must be nonnegative.")
        if sp_ratio <= 0 or sp_ratio >= 1:
            raise ValueError("sp_ratio must be between 0 and 1.")

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.sp_ratio = sp_ratio
        self.sp_type = sp_type
        self.trainable_scaling = trainable_scaling
        self.parameterized_modules = []

        self._config = SpLoRaConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            sp_ratio=sp_ratio,
            sp_type=sp_type,
            target_modules=target_modules,
            random_subspace=random_subspace,
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
            new_module = SpLoRaLinear(
                module.in_features,
                module.out_features,
                r=self.r,
                sp_ratio=sp_ratio,
                sp_type=sp_type,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                trainable_scaling=self.trainable_scaling,
                random_subspace=random_subspace,
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
        #self.wrapped_model.save_pretrained(path)
        self.wrapped_model.save_pretrained(path, safe_serialization=False)
        with open(os.path.join(path, "splora_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        # TODO
        with open(os.path.join(path, "splora_config.json"), "r") as f:
            splora_config = json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)
        if "keep_original" in splora_config:
            print("WARNING: keep_original is deprecated. Use lora_only instead.")
            print(f"keep_original: {splora_config['keep_original']}")
            splora_config["lora_only"] = not splora_config.pop("keep_original")
            splora_config["keep_original_weights"] = not splora_config["lora_only"]

        if "trainable_scaling" not in splora_config:
            splora_config["trainable_scaling"] = False

        model = cls(base_model, **splora_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model


class lora_sparse_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lora_B, lora_A, dv, di, bias):
        ctx.save_for_backward(input, lora_B, lora_A, dv, di, bias)

        return SparseLinearME.apply(input, lora_B, lora_A, dv, di, bias)

    @staticmethod
    def backward(ctx, output_grad):
        input, lora_B, lora_A, dv, di, bias = ctx.saved_tensors

        grads = SparseLinearME.backward(ctx, output_grad)

        return grads


class SpLoRaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        sp_ratio: float = 0.01,
        sp_type: str = "random",
        *,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        trainable_scaling: bool = False,
        random_subspace: bool = False,
        bias=True,
        device=None,
        dtype=None,
    ):
        """
        Reparameterized sparse and low rank linear layer
                    x W_a @ W_b * lora_alpha / r + x W_sp + bias
        Notice that scale = lora_alpha / r.
        Notice that this class cannot be wrapped to linear layer and thus cannot be used for fine-tune
        For fine-tune, please refer to ... TODO
        """
        super().__init__()
        # nn.Module.__init__(self)
        if r <= 0:
            raise ValueError("r must be positive.")
        if sp_ratio <= 0 or sp_ratio >= 1:
            raise ValueError("sp_ratio must be between 0 and 1.")

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
        # self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.random_subspace = random_subspace
        self.trainable_scaling = trainable_scaling
        self.sp_ratio = sp_ratio
        self.sp_type = sp_type
        self.device = device
        self.dtype = dtype

        lora_A_requires_grad = False if random_subspace else True
        self.lora_A = nn.Parameter(
            torch.empty(r, in_features, dtype=dtype, device=device),
            requires_grad=lora_A_requires_grad,
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(
            torch.empty(out_features, r, dtype=dtype, device=device)
        )
        nn.init.zeros_(self.lora_B)
        if trainable_scaling:
            self.scaling = nn.Parameter(
                torch.tensor([1.0], device=device, dtype=dtype), requires_grad=True
            )
        else:
            self.scaling = self.lora_alpha / self.r

        if sp_type.lower() == "random":
            indices, values, shape = self._init_sparse_parameters()
            self.shape = shape
            self.register_buffer("sparse_index", indices.to(device))
            # self.sparse_index =
            self.sparse_value = Parameter(values.to(device), requires_grad=True)

    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()

        return self.scaling

    def _init_sparse_parameters(self):
        # Calculate total elements and the number of non-zero elements
        shape = [self.out_features, self.in_features]
        total_elements = self.in_features * self.out_features
        num_nonzeros = int(self.sp_ratio * total_elements)

        # Generate random indices for non-zero elements
        indices = torch.randperm(total_elements)[:num_nonzeros]
        indices, _ = torch.sort(indices)
        indices.to(self.device)

        # Generate random values for non-zero elements
        values = torch.empty(size=(num_nonzeros,), device=self.device, dtype=self.dtype)
        a = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(values, -a, a)

        return indices, values, shape

    def forward(self, x: Tensor):
        """
        Input x : [..., in_dim] and Output [..., out_dim]
        """
        # out = sp_batch_mm(self.sparse_weight, x) + self.bias
        out = 0
        if self.sp_type.lower() == "random":
            # out += LoraSparseLinear.apply(x, self.lora_B.mm(self.lora_A) * self._post_lora_scale(),
            #                     self.sparse_value, self.sparse_index,
            #                     self.bias)

            # out += sparse_linear.apply(x, self.lora_B.mm(self.lora_A) * self._post_lora_scale(),
            #                      self.sparse_value, self.sparse_index,
            #                      self.bias)
            out += lora_sparse_linear.apply(
                x,
                self.lora_B,
                self.lora_A * self._post_lora_scale(),
                self.sparse_value,
                self.sparse_index,
                self.bias,
            )
            
            # ## if in_features == out_features, add residual connection
            # if self.in_features == self.out_features:
            #     out += x
            
            
            

        return out
    

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, rank={self.r}, "
            f"sparsity={self.sp_ratio}, bias={self.bias is not None}"
        )

    
    @torch.no_grad()
    def merge_and_reinit(self):
        device = self.lora_A.device
        dtype = self.lora_A.dtype  
        
        lora_A = self.lora_A.detach()
        lora_B = self.lora_B.detach()
        
        ## ---------------------------------------------------------------
        # Step 1: Compute W = BA
        W_target = lora_B @ lora_A
        
        W_target_float = W_target.float()
        
        # Step 2: Perform SVD decomposition: W = UΣV^T
        U, S, Vh = torch.linalg.svd(W_target_float, full_matrices=False)  # U, S (singular values), Vh (V^T)

        # Step 3: Keep only the top-r singular values and vectors
        U_r = U[:, :self.r].to(dtype)  # Top-r left singular vectors
        S_r = S[:self.r].to(dtype)     # Top-r singular values
        Vh_r = Vh[:self.r, :].to(dtype) # Top-r right singular vectors

        # Step 4: Reinitialize B and A
        # B = U_r * sqrt(Σ_r)
        # A = sqrt(Σ_r) * Vh_r
        sqrt_S_r = torch.sqrt(S_r)
        new_B = torch.matmul(U_r, torch.diag(sqrt_S_r))  # U_r * sqrt(Σ_r)
        new_A = torch.matmul(torch.diag(sqrt_S_r), Vh_r)  # sqrt(Σ_r) * Vh_r

        # Step 5: Assign new values to lora_A and lora_B
        self.lora_B.copy_(new_B)
        self.lora_A.copy_(new_A)
        ## ---------------------------------------------------------------
        
    

def sparse_linear_forward(input, lora_B, lora_A, dv, di, bias=None):
    device = input.device
    W = lora_B.to(device).mm(lora_A.to(device, lora_B.dtype))
    W.view(-1).scatter_add_(0, di.to(device, torch.int64), dv.to(device, W.dtype))
    return torch.nn.functional.linear(
        input, W.to(device), None if bias is None else bias.to(device)
    )


def sparse_linear_backward(
    output_grad,
    input,
    lora_B,
    lora_A,
    dv,
    di,
    input_needs_grad,
    lora_B_needs_grad,
    lora_A_needs_grad,
    dv_needs_grad,
    bias_needs_grad,
    bias=None,
):
    device = input.device
    W = lora_B.to(device).mm(lora_A.to(device, lora_B.dtype))
    di = di.to(device, torch.int64)
    W.view(-1).scatter_add_(0, di, dv.to(device, W.dtype))

    output_grad_2d = output_grad.reshape(-1, output_grad.size(-1)).to(device)
    input_2d = input.view(-1, input.size(-1)).to(device)

    input_grad = (
        output_grad_2d.mm(W.to(output_grad_2d.dtype)).view_as(input)
        if input_needs_grad
        else None
    )
    lora_A_grad = (
        lora_B.t().to(device).mm(output_grad_2d.t().mm(input_2d))
        if lora_A_needs_grad
        else None
    )
    lora_B_grad = (
        output_grad_2d.t().mm(input_2d).mm(lora_A.t().to(device))
        if lora_B_needs_grad
        else None
    )

    dv_grad = None
    if dv_needs_grad:
        weight_grad = output_grad_2d.t().mm(input_2d.to(output_grad_2d.dtype))
        dv_grad = weight_grad.view(-1).gather(0, di)

    bias_grad = output_grad_2d.sum(0) if bias is not None and bias_needs_grad else None

    return input_grad, lora_B_grad, lora_A_grad, dv_grad, None, bias_grad


class SparseLinearME(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lora_B, lora_A, dv, di, bias=None):
        device = input.device
        if bias is not None:
            ctx.save_for_backward(input, lora_B, lora_A, dv, di, bias)
        else:
            ctx.save_for_backward(input, lora_B, lora_A, dv, di)
        return sparse_linear_forward(
            input,
            lora_B.to(device),
            lora_A.to(device),
            dv.to(device),
            di.to(device),
            bias.to(device) if bias is not None else None,
        )

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        input = saved[0]
        lora_B = saved[1]
        lora_A = saved[2]
        dv = saved[3]
        di = saved[4]
        bias = saved[5] if len(saved) > 5 else None
        output_grad = grad_output

        grads = sparse_linear_backward(
            output_grad,
            input,
            lora_B,
            lora_A,
            dv,
            di,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[3],
            len(saved) > 5 and ctx.needs_input_grad[5],
            bias,
        )

        return grads[0], grads[1], grads[2], grads[3], grads[4], grads[5]


def apply_sparse_linear(input, lora_B, lora_A, dv, di, bias=None):
    return SparseLinearME.apply(input, lora_B, lora_A, dv, di, bias)
