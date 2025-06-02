# Scalable Parameter and Memory Efficient Pretraining for LLM: Recent Algorithmic Advances and Benchmarking

<a href="https://arxiv.org/abs/2505.22922">
  <img src="https://img.shields.io/static/v1?label=arXiv&message=2505.22922&color=b31b1b" />
</a>

This repository contains the code accompanying our paper [Scalable Parameter and Memory Efficient Pretraining for LLM: Recent Algorithmic Advances and Benchmarking](https://arxiv.org/abs/2505.22922).  It includes implementations of various parameter and memory efficient pretraining methods for large language models. It provides a unified framework for experimenting with different optimization algorithms and techniques for efficient pretraining.

## Installation

```bash
cd parameter_efficient_pretraining
pip install -r requirements.txt
pip install -e .
```

## Getting Started

The repository provides an example training script in the `scripts/` directory. To start training, you can use this script as a template and modify it for your specific needs.

For example, to train a 60M parameter model with low-rank optimization:

```bash
bash scripts/60m_low_rank.sh
```

## Repository Structure

- **configs/**: Configuration files for different model sizes (9M to 13B parameters)
  - Support for Llama and Qwen2 architectures

- **para_eff_pt/**: Core package containing implementations of all parameter-efficient training methods
  - Each submodule (`pt_*`) implements a specific algorithm

- **scripts/**: Contains an example script that you can use as a template for your own training configurations

- **torchrun_main_DDP.py**: Main entry point for distributed training

## Supported Algorithms

This repository implements the following training methods:

1. **AdamW**
2. **Apollo**
3. **FIRA**
4. **Galore**
5. **Golore**
6. **LoRO**
7. **Low-Rank**
8. **ReLoRA**
9. **SLTrain**
10. **SPAM**
11. **Stable SPAM**

## Usage

The repository provides an example script (`60m_low_rank.sh`) that you can use as a reference for creating your own training configurations. You'll need to configure parameters specific to your chosen algorithm and model size.

Example script (`60m_low_rank.sh`):

```bash
torchrun --standalone --nproc_per_node 4 torchrun_main_DDP.py \
    --model_name low-rank \
    --model_config configs/llama_60m.json \
    --lr 0.0015 \
    --peft_model low-rank \
    --optimizer adamw \
    --rank 128 \
    --lora_alpha 32 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 11000 \
    --warmup_steps 1100 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000
```

Key parameters:
- `--nproc_per_node`: Number of GPUs to use (4 in this example)
- `--model_name`: The type of model to use
- `--model_config`: Path to the model configuration file
- `--peft_model`: Parameter-efficient training method
- `--rank`: Rank for low-rank methods
- `--batch_size`: Batch size per GPU
- `--total_batch_size`: Total batch size across all GPUs
- `--num_training_steps`: Total number of training steps
- `--dtype`: Training precision (bfloat16 in this example)

## Configuration

The `configs/` directory contains JSON configuration files for different model sizes. For example, `llama_60m.json` defines the architecture for a 60M parameter Llama model.

To use a different model size, simply change the configuration file in your script:

```bash
--model_config configs/llama_1b.json  # Use 1B parameter model
```

To experiment with different algorithms, modify the `--peft_model` parameter to one of the supported algorithms and configure any algorithm-specific parameters as needed.

### Citation

If you find this work useful for your research, please cite our paper:
```bibtex
@article{glentis2025scalable,
  title={Scalable Parameter and Memory Efficient Pretraining for LLM: Recent Algorithmic Advances and Benchmarking},
  author={Glentis, Athanasios and Li, Jiaxiang and Shang, Qiulin and Han, Andi and Tsaknakis, Ioannis and Wei, Quan and Hong, Mingyi},
  journal={arXiv preprint arXiv:2505.22922},
  year={2025}
}
```
