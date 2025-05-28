import torch
import torch.nn as nn
import transformers
import bitsandbytes as bnb

from para_eff_pt.peft_pretraining import training_utils
from para_eff_pt.pt_low_rank import LoRaFaModel
from para_eff_pt.pt_sltrain import SpLoRaModel

from para_eff_pt.pt_relora import ReLoRaModel
from para_eff_pt.pt_galore import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

from para_eff_pt.pt_golore import GoLoreAdamW,GoLoreAdamW8bit,GoLoreSGD
from para_eff_pt.pt_golore import GoloreReLoRaModel,GoloreReLoRaLinear
from para_eff_pt.pt_loro import LORO_optimizer

from para_eff_pt.pt_spam import SPAM_optimizer 

from para_eff_pt.pt_stable_spam import StableSPAM_optimizer

from para_eff_pt.pt_fira import Fira_AdamW

from para_eff_pt.pt_apollo import Apollo_AdamW


def build_model(model, args):
    if args.peft_model.lower() == "low-rank":
        model = LoRaFaModel(
            model,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            trainable_scaling=args.train_scaling,
        )
    elif args.peft_model.lower() == "sltrain":
        model = SpLoRaModel(
            model,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            trainable_scaling=args.train_scaling,
            sp_ratio=args.sp_ratio,
            sp_type=args.sp_type,
            random_subspace=args.random_subspace,
        )

    elif args.peft_model.lower() == "relora":
        model = ReLoRaModel(
            model,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            trainable_scaling=args.train_scaling,
        )

 
    elif args.peft_model.lower() == "golore":
        model = GoloreReLoRaModel(
            model,
            r=args.rank,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )
    

    return model



def build_optimizer(model, trainable_params, args):   
    #if args.optimizer.lower() == "adamw_beta":
    #    optimizer = adamw_beta(trainable_params, lr=args.lr, weight_decay=args.weight_decay, cycle_length=args.cycle_length)
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.beta1,
        )
    # implement adafactor
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "adam8bit_per_layer":
        optimizer = {}
        for p in model.parameters():
            if p.requires_grad:
                optimizer[p] = bnb.optim.Adam8bit(
                    [p], lr=args.lr, weight_decay=args.weight_decay
                )
        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None:
                return
            optimizer[p].step()
            optimizer[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
            
    
    elif args.optimizer.lower() == "loro_optimizer":
        optimizer = LORO_optimizer(
            trainable_params, lr=args.lr, K=10
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    return optimizer



def build_optimizer_apollo(model, param_groups, id_galore_params, args):
    if args.optimizer.lower() == "apollo_adamw":
        optimizer = Apollo_AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, scale_front=args.scale_front, disable_nl=args.disable_nl)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    return optimizer 


def build_optimizer_fira(model, param_groups, id_galore_params, args):
    if args.optimizer.lower() == "fira_adamw":
        optimizer = Fira_AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, disable_nl=args.disable_nl)
        print("using fira optimizer !!")
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    return optimizer 
    
    

def build_optimizer_galore(model, param_groups, id_galore_params, args):
    layer_wise_flag = False
    if args.optimizer.lower() == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(
            param_groups, lr=args.lr, weight_decay=args.weight_decay, galore_use_nl=args.galore_use_nl
        )
    # low-rank adafactor
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif args.optimizer.lower() == "galore_adamw8bit":
        optimizer = GaLoreAdamW8bit(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "galore_adamw8bit_per_layer":
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit(
                        [
                            {
                                "params": [p],
                                "rank": args.rank,
                                "update_proj_gap": args.update_proj_gap * 2,
                                "scale": args.galore_scale,
                                "proj_type": args.proj_type,
                            }
                        ],
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                    )
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit(
                        [p], lr=args.lr, weight_decay=args.weight_decay
                    )

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None:
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        layer_wise_flag = True

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    return optimizer

def build_optimizer_golore(model, param_groups, args):
    layer_wise_flag = False
    if args.optimizer.lower() == "golore_adamw":

        optimizer = GoLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "golore_adamw8bit":
 
        optimizer = GoLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "golore_sgd":
     
        optimizer = GoLoreSGD(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        
    return optimizer


def build_optimizer_spam(model, param_groups, args):
    if args.optimizer.lower() == "spam_adamw":
        optimizer = SPAM_optimizer(param_groups, lr=args.lr,weight_decay=args.weight_decay,warmup_epoch=args.warmup_epoch,threshold=args.threshold)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
        
    return optimizer

def build_optimizer_stable_spam(model, param_groups, args):
    if args.optimizer.lower() == "stable_spam_adamw":
        optimizer = StableSPAM_optimizer(params=param_groups, lr = args.lr, weight_decay = args.weight_decay,gamma1=args.gamma1,gamma2=args.gamma2,gamma3=args.gamma3,eta_min=args.eta,update_proj_gap=args.update_gap,total_T=args.total_T)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
        
    return optimizer


def build_optimizer_loro(model, param_groups, args):
    """
    Build the LORO optimizer
    """
    if args.optimizer.lower() == "loro_optimizer":
        regular_group = param_groups[0]
        loro_group = param_groups[1]
        
        print(f"Regular params: {len(regular_group['params'])}")
        print(f"LORO params: {len(loro_group['params'])}")
        
        # Create AdamW optimizer for regular parameters
        adamw_optimizer = torch.optim.AdamW(
            regular_group['params'],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Create LORO optimizer for low-rank parameters
        loro_optimizer = LORO_optimizer(
            [loro_group],  # Pass the entire parameter group here
            lr=args.lr,
            weight_decay=args.weight_decay,
            update_k=args.cycle_length,
        )
        
        ## To test if the structure is correct
        # loro_adam_optimizer = torch.optim.AdamW(
        #     loro_group['params'],
        #     lr=args.lr,
        #     weight_decay=args.weight_decay
        # )
        
        class CombinedOptimizer(torch.optim.Optimizer):
            def __init__(self, adamw_opt, loro_opt, loro_adam_optimizer=None):
                self.adamw = adamw_opt
                self.loro = loro_opt
                self.loro_adam = loro_adam_optimizer
                self.step_count = 0
                
                # Collect all parameters
                params = []
                param_groups = []
                for group in self.adamw.param_groups:
                    params.extend(group['params'])
                    param_groups.append(group)
                for group in self.loro.param_groups:
                    params.extend(group['params'])
                    param_groups.append(group)
                # for group in self.loro_adam.param_groups:
                #     params.extend(group['params'])
                #     param_groups.append(group)
                    
                # Call the parent class constructor
                defaults = {
                    'lr': param_groups[0]['lr'],
                    'weight_decay': param_groups[0]['weight_decay']
                }
                super().__init__(params, defaults)
                
                # Restore the original param_groups
                self.param_groups = param_groups
                
            def zero_grad(self, set_to_none: bool = False):
                self.adamw.zero_grad(set_to_none=set_to_none)
                self.loro.zero_grad(set_to_none=set_to_none)
                # self.loro_adam.zero_grad(set_to_none=set_to_none)
                
            @torch.no_grad()
            def step(self, closure=None):
                # Update learning rates for each optimizer before stepping
                adamw_groups_len = len(self.adamw.param_groups)
                
                for i, group in enumerate(self.adamw.param_groups):
                    group['lr'] = self.param_groups[i]['lr']
                
                for i, group in enumerate(self.loro.param_groups):
                    group['lr'] = self.param_groups[i + adamw_groups_len]['lr']
                
                # for i, group in enumerate(self.loro_adam.param_groups):
                #     group['lr'] = self.param_groups[i + adamw_groups_len]['lr']
                    
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
        
                self.adamw.step()
         
                self.loro.step()
                # self.loro_adam.step()
                
                if self.loro.is_exact == True:
                    print(f"Resetting optimizer state after exact update at step {self.step_count + 1}")
                    
                    ## Reset AdamW optimizer
                    self.adamw = torch.optim.AdamW(
                        regular_group['params'],
                        lr=args.lr,
                        weight_decay=args.weight_decay
                    )
                    
                    ## Reset LORO optimizer
                    self.loro = LORO_optimizer(
                        [loro_group],  # Pass the entire parameter group here
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        update_k=args.cycle_length,
                    )
                    
                    self.loro.is_exact = False

                self.step_count += 1
                
                return loss
                
            def state_dict(self):
                return {
                    'adamw': self.adamw.state_dict(),
                    'loro': self.loro.state_dict(),
                    'param_groups': self.param_groups,
                    'state': self.state,
                }
                
            def load_state_dict(self, state_dict):
                self.adamw.load_state_dict(state_dict['adamw'])
                self.loro.load_state_dict(state_dict['loro'])
                self.param_groups = state_dict['param_groups']
                self.state = state_dict['state']
        
        optimizer = CombinedOptimizer(adamw_optimizer, loro_optimizer)
        
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    
    return optimizer
