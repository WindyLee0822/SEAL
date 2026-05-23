from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset, load_from_disk
from string import Template
import argparse
import torch.distributed as dist
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    return parser.parse_args()

def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_path = './data/OpenManus-RL/webshop_train_sft.parquet'
    test_path = './data/OpenManus-RL/webshop_test_sft.parquet'
    train_dataset = Dataset.from_parquet(train_path)

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own.
""")

        return {
            "prompt": [{"role": "user", "content": example['messages'][0]['content']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['messages'][0]['content'], 
                                                                                     output_text='\n'.join(example['messages'][1]['content']))}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


if __name__ == "__main__":
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset, _ = load_tooluse_dataset(args.seed)

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        # vllm_mode="colocate",
        # vllm_tensor_parallel_size=1, 
        # vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True, 
        vllm_mode="server",   
        vllm_server_base_url="http://0.0.0.0:8000",
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = args.num_prompts_per_batch,
        max_prompt_length = 4096,
        max_completion_length = 1024,
        num_train_epochs = args.num_train_epochs,
        save_steps = 100,
        max_grad_norm = 1,
        report_to = "wandb",
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        sync_ref_model = True,
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction = True,
        num_loss_tokens_to_skip = 3,
    )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()