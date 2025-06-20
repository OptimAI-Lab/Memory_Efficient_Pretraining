import os
import time
import json
import random

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from safetensors.torch import load_file

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
import datasets
import datasets.distributed

import wandb
from tqdm import tqdm
from loguru import logger

from para_eff_pt.peft_pretraining import training_utils
from para_eff_pt.peft_pretraining.dataloader import PreprocessedIterableDataset, PreprocessedIterableDataset_noslice


from para_eff_pt.peft_pretraining.dataloader_v2 import PreprocessedIterableDataset_v2


from para_eff_pt.peft_pretraining.modeling_llama import LlamaForCausalLM


from para_eff_pt.pt_sltrain import *
from para_eff_pt.pt_low_rank.low_rank_model import *

from para_eff_pt.utils.train_utils import *
from para_eff_pt.utils.args import parse_args
 
 

transformers.logging.set_verbosity_error()
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)





@torch.no_grad()
def evaluate_model(
    model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size
):
    _time = time.time()
    if not args.hf_dataset:
        logger.info(f"Using local dataset for validation")
        data_files_val= {"validation": [f"{args.dataset_path}/c4-validation.{str(i).zfill(5)}-of-00008.json.gz" for i in range(0,8)]}
        val_data = datasets.load_dataset(path=args.dataset_path,  data_files=data_files_val, split="validation", streaming=True)
    else:
        val_data = datasets.load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        )  # DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(
            val_data, rank=global_rank, world_size=world_size
        )

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(
        val_data_mapped, batch_size
    )

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(
        f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}"
    )

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert (
                args.total_batch_size % world_size == 0
            ), "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (
                args.batch_size * world_size
            )
            # logger.info(f"{args.gradient_accumulation}-{world_size}-{args.total_batch_size}-{args.batch_size}")
            assert (
                args.gradient_accumulation > 0
            ), "gradient_accumulation must be greater than 0"

    assert (
        args.gradient_accumulation * args.batch_size * world_size
        == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0:
        logger.remove()

    if global_rank == 0:
        wandb.init(project=args.wandb_project_name, name=args.model_name)

        logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
        logger.info("*" * 40)
        logger.info(f"Starting training with the arguments")
        for k, v in vars(args).items():
            logger.info(f"{k:30} {v}")
        logger.info("*" * 40)

    # data
    if not args.hf_dataset:
        logger.info(f"Using local dataset for training")
        data_files_train = {"train": [f"{args.dataset_path}/c4-train.{str(i).zfill(5)}-of-01024.json.gz" for i in range(0,1024)]}
        logger.info(f"loading dataset")
        data = datasets.load_dataset(path=args.dataset_path,  data_files=data_files_train, split="train", streaming=True)
        logger.info(f"loaded dataset")
    else:
        data = datasets.load_dataset(
            "allenai/c4", "en", split="train", streaming=True
        )  # DGX


    seed_for_shuffle = 42
    

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data,
            rank=global_rank,
            world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", model_max_length=args.max_length
    )

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

 
    if args.continue_from is not None:
 
        dataset = PreprocessedIterableDataset_v2(
            data, tokenizer, batch_size=args.batch_size, max_length=args.max_length, start_tokenizing_idx = args.start_tokenizing_idx
        )
    else:
        if args.no_slice: # !!!
            logger.info(f"Using PreprocessedIterableDataset_noslice !!")
            dataset = PreprocessedIterableDataset_noslice(
                data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
            )
        else:
             dataset = PreprocessedIterableDataset(
                data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
            )           



    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=args.workers
    )
 

    # model
    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    # ====== starting config ======= #
    target_modules_list = ["attn", "mlp", "attention"]
    args.target_modules = target_modules_list

    # build model
    if args.dtype in ["bf16", "bfloat16"]:
        model = build_model(model.to(device=device, dtype=torch.bfloat16), args)
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = build_model(model.to(device=device), args)
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if 'spam' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        spam_params = []
        target_modules_list = ["attn", "mlp","attention"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            spam_params.append(module.weight)
        id_spam_params = [id(p) for p in spam_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_spam_params]

        # then call spam_adamw
        param_groups = [{'params': regular_params}, 
                        {'params': spam_params, 'density': args.density, 'update_proj_gap': args.update_gap}]

    if ("galore" in args.optimizer.lower()) or ("fira" in args.optimizer.lower()):
        
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            
            
            if not ((isinstance(module, nn.Linear)) or (module.__class__.__name__ == 'SpLoRaLinear') or (module.__class__.__name__=='Restart_LoRaLinear')  ):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print(f"enable {args.optimizer.lower()} for weights in module: ", module_name)
            
            if module.__class__.__name__ == 'SpLoRaLinear' or module.__class__.__name__ == 'Restart_LoRaLinear' :
                galore_params.append(module.lora_A)  
                
            else:
                galore_params.append(module.weight)
            
            
            
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [
            p for p in model.parameters() if id(p) not in id_galore_params
        ]
        
        opt_rank = args.rank
        if args.opt_rank != -1:
            opt_rank = args.opt_rank
        
        
        # then call galore_adamw
        param_groups = [
            {"params": regular_params},
            {
                "params": galore_params,
                "rank": opt_rank,
                "update_proj_gap": args.update_proj_gap,
                "scale": args.galore_scale,
                "proj_type": args.proj_type,
            },
        ]
    elif "apollo" in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        lowrank_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            #if not (isinstance(module, nn.Linear)):
            if not ((isinstance(module, nn.Linear)) or (module.__class__.__name__ == 'SpLoRaLinear') or (module.__class__.__name__=='Restart_LoRaLinear')  ) :    
            #if not (isinstance(module, nn.Linear) or isinstance(module, QScaleLinear) or isinstance(module, QLinear)):  
                continue
            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            logger.info(f"Adding {module_name} to APOLLO parameters")
            
            if module.__class__.__name__ == 'SpLoRaLinear' or module.__class__.__name__ == 'Restart_LoRaLinear':
                lowrank_params.append(module.lora_A)
            else:
                lowrank_params.append(module.weight)

        id_lowrank_params = [id(p) for p in lowrank_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_lowrank_params]
        # then call low rank optimizer
        param_groups = [
            {"params": regular_params},
            {
                "params": lowrank_params,
                "rank": args.rank,
                "update_proj_gap": args.update_proj_gap,
                "scale": args.apollo_scale,
                "proj_type": args.proj_type,
                "proj": args.proj,
                "scale_type": args.scale_type,
            },
        ]


    # build optimizer

    if "galore" in args.optimizer.lower():
        optimizer = build_optimizer_galore(model, param_groups, id_galore_params, args)
    elif "fira" in args.optimizer.lower():
        optimizer = build_optimizer_fira(model, param_groups, id_galore_params, args)
    elif "apollo" in args.optimizer.lower():
        optimizer = build_optimizer_apollo(model, param_groups, id_lowrank_params, args)
    elif 'stable_spam' in args.optimizer.lower():
        optimizer = build_optimizer_stable_spam(model, param_groups,  args)  
    elif 'spam' in args.optimizer.lower():
        optimizer = build_optimizer_spam(model, param_groups,  args)  
    elif "golore" in args.optimizer.lower():
        optimizer = build_optimizer_golore(model, trainable_params, args)
        
    elif "loro" in args.optimizer.lower():
            # Identify low-rank parameters
            loro_params = []
            regular_params = []
            processed_params = set()  # Used to track already processed parameters
            
            for module_name, module in model.named_modules():
                if isinstance(module, LoRaFaLinear):
                    print("Enable LORO for weights in module: ", module_name)
                    # Directly use lora_A and lora_B modules instead of their weights
                    loro_params.append({
                        'A': module.lora_A,
                        'B': module.lora_B,
                        'name': module_name
                    })
                    # Add processed parameter IDs
                    processed_params.add(id(module.lora_A.weight))
                    processed_params.add(id(module.lora_B.weight))
                    if module.bias is not None:
                        processed_params.add(id(module.bias))
            
            # Collect other parameters
            for name, param in model.named_parameters():
                if param.requires_grad and id(param) not in processed_params:
                    regular_params.append(param)
                    processed_params.add(id(param))
            
            # Print parameter statistics for debugging
            print(f"Found {len(loro_params)} LORO parameter pairs")
            print(f"Found {len(regular_params)} regular parameters")
            
            # Create optimizer group for LORO parameters
            loro_optimizer_params = []
            for param_dict in loro_params:
                loro_optimizer_params.extend([
                    param_dict['A'].weight,
                    param_dict['B'].weight
                ])
            
            param_groups = [
                {"params": regular_params},
                {
                    "params": loro_optimizer_params,
                    "update_k": args.cycle_length,
                    "is_loro": True
                }
            ]
            
            optimizer = build_optimizer_loro(model, param_groups, args)
            
     
        
    else:
        optimizer = build_optimizer(model, trainable_params, args)
       

    layer_wise_flag = True if "per_layer" in args.optimizer.lower() else False
    if layer_wise_flag:
        if not isinstance(optimizer, dict):
            raise ValueError("Layer-wise optimizer is not properly constructed.")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            cycle_length=args.cycle_length,
            recovery_steps=args.recovery_steps,

        )

    if args.continue_from is not None:

        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        
        # !!!!
        if not os.path.exists(checkpoint_path): #safetensors -> bin  
            safetensors_file = os.path.join(args.continue_from, "model.safetensors")
            state_dict = load_file(safetensors_file)
            torch.save(state_dict, checkpoint_path)
 
            logger.info(f"safetensors {safetensors_file} converted to pytorch bin {checkpoint_path}")
        
        
        if args.peft_model.lower() in ["sltrain"]:
            model.wrapped_model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu"), strict=True
            )
        else:
            model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu"), strict=True
            )
        logger.info(f"Model successfully loaded (strict=True policy)")

        optimizer_checkpoint = torch.load(
            os.path.join(args.continue_from, "optimizer.pt"), map_location="cpu"
        )
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        logger.info(f"Optimizer and scheduler restored from {args.continue_from}")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}"
            )
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(
                f"Will train for {args.num_training_steps - update_step} update steps"
            )
        else:
            logger.warning(
                f"Did not find training state in {args.continue_from}, global step will start from zero"
            )
        logger.info("*" * 40)

    scheduler_start_step = update_step


    # print params and trainable params
    logger.info(f"Running with {args.peft_model}\n")
    logger.info(f"\n{model}\n")
    logger.info(
        f"All params: \n{[n for n,p in model.named_parameters() if p.requires_grad]}\n"
    )
    logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M"
    )
    logger.info(
        f"Total non-low-rank and non-sparse parameters: "
        f"{sum(p.numel() for n,p in model.named_parameters() if 'lora_' not in n and 'sparse_' not in n) / 1_000_000:.2f}M"
    )

    if args.peft_model.lower() == "sltrain":
        logger.info(
            f"Total low-rank parameters: "
            f"{sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n) / 1_000_000:.2f}M"
        )
        logger.info(
            f"Total low-rank parameters (requires_grad): "
            f"{sum(p.numel() for n,p in model.named_parameters() if 'lora_' in n and p.requires_grad) / 1_000_000:.2f}M"
        )
        logger.info(
            f"Total sparse parameters: "
            f"{sum(p.numel() for n, p in model.named_parameters() if 'sparse_' in n and p.requires_grad) / 1_000_000:.2f}M"
        )

    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    )
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update(
        {
            "max_lr": run_config.pop(
                "lr"
            ),  # rename lr to max_lr to avoid conflicts with scheduler
            "total_params_M": n_total_params / 1_000_000,
            "dataset": "allenai/c4",
            "model": model_config.to_dict(),
            "world_size": world_size,
            "device": str(device),
        }
    )

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")  # save current script
        pbar = tqdm(
            total=args.num_training_steps - update_step, desc="Update steps", ncols=80
        )

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,    
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # ##############################

    grad_norm_prev = None

    max_memory = torch.cuda.max_memory_allocated()
    if global_rank == 0:
        logger.info(f"Maximum memory allocated before training: {max_memory} bytes\n")
    torch.cuda.reset_peak_memory_stats()


    boo = False
     
    for batch_idx, batch in enumerate(dataloader):

        if args.continue_from is not None and not boo:   
            if batch_idx   <=   (update_step) * args.gradient_accumulation  - 1 : 
                if batch_idx % 1000 == 0:
                    print(batch_idx)
                continue
            else:
                print(f"\n start at {batch_idx} \n")
                boo = True
                

        if update_step == 0 and args.eval_at_begining :
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model,
                preprocess_batched,
                pad_idx,
                global_rank,
                world_size,
                device,
                args.batch_size,
            )
            if global_rank == 0:
                wandb.log(
                    {
                        "final_eval_loss": total_loss,
                        "final_eval_perplexity": np.exp(total_loss),
                        "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(
                f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
            )        


        global_step += 1
        local_step += 1
        

        if update_step > args.num_training_steps:
            logger.info(
                f"Reached max number of update steps (f{args.num_training_steps}). Stopping training."
            )
            print(f"Rank {global_rank} stopping training.")
            break
        
        if args.peft_model.lower() == 'golore':
            reset_relora = update_step % args.update_proj_gap == 0
            unwrapped_model = model.module if hasattr(model, 'module') else model
            unwrapped_model._config.forward_type = reset_relora

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size


        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()    

        if global_step % args.gradient_accumulation != 0:
            continue

        if args.grad_clipping != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        grad_norm = sum(
            [
                torch.norm(p.grad.clone().detach().cpu())
                for p in model.parameters()
                if p.grad is not None
            ]
        )
        
           
        if args.rand_ratio > 0.0 and args.peft_model.lower() == 'golore' and reset_relora:
            _lora_reset_time = time.time()
            # logger.info(f"{args.resume_from=}, {local_step=}, {args.relora=}, thresh: {local_step // args.gradient_accumulation}")
            logger.info(f"Performing lora reset at update step {update_step}. Current lr is {optimizer.param_groups[0]['lr']}")

            use_rand = update_step / args.num_training_steps >= args.rand_ratio
            
            underlying_model = model.module if hasattr(model, 'module') else model
            
            underlying_model.merge_and_reinit(optimizer,rand=use_rand)
            
            
        if global_rank == 0:
            pbar.update(1)

        if not layer_wise_flag:
            optimizer.step()    
            scheduler.step()
            optimizer.zero_grad()
            
        
        #print(gradnorms)

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if (
            local_step > args.gradient_accumulation
            and update_step % args.save_every == 0
            and global_rank == 0
        ):
            if args.keep_only_last_model:
                current_model_directory = f"{args.save_dir}/model_last"
            else:
                current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(
                f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
            )
            os.makedirs(args.save_dir, exist_ok=True)
            model.module.save_pretrained(
                current_model_directory, max_shard_size="100GB"
            )

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model,
                preprocess_batched,
                pad_idx,
                global_rank,
                world_size,
                device,
                args.batch_size,
            )
            if global_rank == 0:
                wandb.log(
                    {
                        "final_eval_loss": total_loss,
                        "final_eval_perplexity": np.exp(total_loss),
                        "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(
                f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
            )
            
        
            
        if update_step % args.cycle_length == 0 and (args.peft_model.lower() == 'sltrain'):
            logger.info(f"\nReinitialize B,A at update step {update_step}")
            
            ## There is a bug with the merge & reinitialize operation on FSDP, and the solution is currently unknown. Use DDP for now.
            '''
            RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet. 
            Caffe2 uses a lazy allocation, so you will need to call mutable_data() or raw_mutable_data() to actually allocate memory.
            '''
                        
            underlying_model = model.module if hasattr(model, 'module') else model
            for name, module in underlying_model.named_modules():   
                if isinstance(module, SpLoRaLinear):
                    module.merge_and_reinit()
                    
            
            new_params = [p for p in model.parameters() if p.requires_grad]
            
            args.lr = args.lr 
            num_training_steps = args.num_training_steps  # Use total steps
            
            
            if "fira" in args.optimizer.lower():
                optimizer = build_optimizer_fira(model, param_groups, id_galore_params, args)
            elif "apollo" in args.optimizer.lower():
                optimizer = build_optimizer_apollo(model, param_groups, id_lowrank_params, args)
            else:
                optimizer = build_optimizer(model, new_params, args)
            
            # Add initial_lr
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = args.lr  

                        
            scheduler = training_utils.get_scheculer(
                    optimizer=optimizer,
                    scheduler_type="cosine_quick_recovery",
                    num_training_steps=num_training_steps,  # Use total steps
                    warmup_steps=args.warmup_steps,  # No impact
                    min_lr_ratio=args.min_lr_ratio,
                    cycle_length=args.cycle_length,  # Restart interval
                    last_epoch=update_step,
                    recovery_steps=args.recovery_steps,
                )


        
        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        max_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        if global_rank == 0:
            wandb.log(
                {
                    "loss": loss.item(),
                    "lr": lr,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "throughput_tokens": tokens_in_update / update_time,
                    "throughput_examples": args.total_batch_size / update_time,
                    "throughput_batches": batches_in_update / update_time,
                    "gradnorm": grad_norm,
                    "max_memory": max_memory,
                },
                step=global_step,
            )
            
                
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0:
        pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(
            f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
        )
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model,
        preprocess_batched,
        pad_idx,
        global_rank,
        world_size,
        device,
        args.batch_size,
    )

    if global_rank == 0:
        wandb.log(
            {
                "final_eval_loss": total_loss,
                "final_eval_perplexity": np.exp(total_loss),
                "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(
            f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
        )

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
