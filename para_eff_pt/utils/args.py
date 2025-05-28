import argparse

import torch

from para_eff_pt.peft_pretraining import args_utils, training_utils


def parse_args(args):
    parser = argparse.ArgumentParser()

    # Hyperparameters for APOLLO
    parser.add_argument("--proj", type=str, default="random") # "random" or "svd"
    parser.add_argument("--scale_type", type=str, default="tensor") # "tensor" or "channel"
    parser.add_argument("--apollo_scale", type=float, default=1.0) # scale for gradient scaling factor
    parser.add_argument("--scale_front", action='store_true') # put the nl before or after scale the gradient with the apollo_scale
    parser.add_argument("--disable_nl", action='store_true') # disables grad clipping (Norm-Growth Limiter)
    
    parser.add_argument("--galore_use_nl", action='store_true') # enables galore grad clipping (Norm-Growth Limiter)
 
    # Hyperparameters for Stable-SPAM
    parser.add_argument("--gamma1", type=float, default=0.85) # beta1 for Adafactor, GaLore_adafactor, (Q-)GaLore-adam or SGD
    parser.add_argument("--gamma2", type=float, default=0.999) # beta2 for (Q-)GaLore-adam
    parser.add_argument("--gamma3", type=float, default=0.999) # beta2 for (Q-)GaLore-adam
    parser.add_argument("--total_T", type=int, default=20000) # beta2 for (Q-)GaLore-adam # ???? total number of update steps ????
    parser.add_argument("--eta", type=float, default=0.5)

    # Hyper-parameters for SPAM
    parser.add_argument("--warmup_epoch", type=int, default=150)
    parser.add_argument("--threshold", type=float, default=5000)
    parser.add_argument("--grad_accu_steps",type=float,default=20)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--update_gap", type=int, default=500)
    

    parser.add_argument(
        "--recovery_steps",
        type=int,
        default=10,
        help="Number of steps for cosine restarts (only used for cosine_restarts)",
    )
    

    parser.add_argument("--eval_at_begining", default=False, action="store_true")
    
    parser.add_argument("--start_tokenizing_idx", type=int, default=0)
    
    parser.add_argument("--no_slice", default=False, action="store_true")
    
    parser.add_argument("--sweep_seed", default=False, action="store_true")

    parser.add_argument("--keep_only_last_model", default=False, action="store_true")
    
    parser.add_argument("--dataset_path", type=str)
    
    parser.add_argument("--hf_dataset", default=False, action="store_true")
    
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--model_name", type=str)

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_restarts","cosine_quick_recovery"],
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=10_000,
        help="Number of **update steps** to train for. "
        "Notice that gradient accumulation is taken into account.",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=training_utils.max_train_tokens_to_number,
        default=None,
        help="Number of tokens to train on. Overwrites num_training_steps. "
        "You can use M and B suffixes, e.g. 100M or 1B.",
    )
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    # ===== additional argument =====
    parser.add_argument(
        "--peft_model",
        type=str,
        default="full-rank",
        choices=[
            "full-rank",
            "low-rank",
            "sltrain",
            "lora",
            "relora",
            "galore",
            "restart_lora",
            "restart_sltrain",
            "golore",
            "fira",
            "apollo",
        ],
    )
    # lore parameters
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--train_scaling", default=False, action="store_true")
    parser.add_argument("--rank", type=int, default=32)

    # relora parameters
    parser.add_argument("--relora", type=int, default=500)
    parser.add_argument(
        "--cycle_length",
        type=int,
        default=65000,
        help="Number of steps per cycle for cosine scheduler",
    )
    parser.add_argument(
        "--restart_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for cosine restarts (only used for cosine_restarts). Not used ?",
    )
    parser.add_argument(
        "--adjust_step",
        type=int,
        default=0,
        help="Number of steps to adjust the scheduler by. "
        f"Useful when you want to sync ReLoRA resets with the scheduler for a warmed up model. "
        f"You need to use it, when your warmup_step % relora_resets != 0",
    )
    parser.add_argument(
        "--reset_optimizer_on_relora", default=True, type=lambda x: x.lower() == "true"
    )
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--optimizer_random_pruning",
        default=0.0,
        type=float,
        help="Use random pruning to reduce optimizer matrix internal dimensionality.",
    )
    parser.add_argument(
        "--optimizer_magnitude_pruning",
        default=0.0,
        type=float,
        help="Use magnitude pruning to reduce optimizer matrix internal dimensionality.",
    )
    parser.add_argument(
        "--distributed_type", type=str, default="ddp", choices=["fsdp", "ddp"]
    )

    # sltrain parameters
    parser.add_argument("--sp_ratio", type=float, default=0.01)
    parser.add_argument("--sp_type", default="random", type=str, choices=["random"])
    parser.add_argument("--random_subspace", default=False, action="store_true")
    parser.add_argument("--precondition", default=False, action="store_true")
    parser.add_argument("--precon_type", default="norm", type=str)
    parser.add_argument("--f_decay", default=0.0, type=float, help="Frobenius decay")

    # GaLore parameters
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")

    # fourier parameters
    parser.add_argument("--n_freq", type=int, default=10_000)
    parser.add_argument("--fourier_scale", type=float, default=20.0)
    
    ## restart parameters
    parser.add_argument("--restart_every", type=int, default=11000)   
    parser.add_argument("--gradient_control", default=False)
    
    ## golore parameters
    parser.add_argument("--rand_ratio", type=float, default=-1.0)

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)

    return args
