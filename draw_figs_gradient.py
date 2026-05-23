import os
import csv
import json
import math
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def global_grad_norm(model):
    total = 0.0

    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm(2).item() ** 2

    return math.sqrt(total)


def response_only_loss(logits, labels):
    """
    Single-example causal LM loss.
    labels should already contain -100 for prompt tokens.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    valid = shift_labels != -100
    num_tokens = valid.sum()

    if num_tokens.item() == 0:
        return None, 0

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )

    loss = loss / num_tokens
    return loss, int(num_tokens.item())


def compute_one_grad_norm(
    model,
    tokenizer,
    prompt,
    response,
    device,
    max_length=4096,
    use_bf16=True,
):
    text = prompt + response

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    prompt_len = len(prompt_ids)

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    # In case truncation removed the response
    if (labels[:, 1:] != -100).sum().item() == 0:
        return None, 0, None

    model.zero_grad(set_to_none=True)

    if use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            loss, num_tokens = response_only_loss(outputs.logits, labels)
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        loss, num_tokens = response_only_loss(outputs.logits, labels)

    if loss is None:
        return None, 0, None

    loss.backward()

    grad_norm = global_grad_norm(model)

    loss_value = float(loss.detach().cpu())
    grad_norm_value = float(grad_norm)

    del outputs, loss
    model.zero_grad(set_to_none=True)

    return loss_value, num_tokens, grad_norm_value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="checkpoints/verl_agent_webshop/grpo-llama3binstructsft_lr2e-6envstep10/global_step_400/hf",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/entropy_analyses/llama3grpo_step400_webshop.json",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="response_grad_norms",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use fp32 instead of bf16. More exact but much more memory.",
    )

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = True

    dtype = torch.float32 if args.fp32 else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.to(device)
    model.eval()

    # Optional memory saver; slower but often helps.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    with open(args.data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.max_examples is not None:
        dataset = dataset[:args.max_examples]

    shard_indices = list(range(rank, len(dataset), world_size))

    output_csv = f"{args.output_prefix}.rank{rank}.csv"

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "prompt",
                "response",
                "loss",
                "num_loss_tokens",
                "grad_norm",
                "correctness",
            ],
        )
        writer.writeheader()

        for idx in tqdm(shard_indices, desc=f"rank {rank}"):
            prompt = dataset[idx][0]
            response = dataset[idx][1]
            correctness = dataset[idx][2]

            loss, num_tokens, grad_norm = compute_one_grad_norm(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                response=response,
                device=device,
                max_length=args.max_length,
                use_bf16=not args.fp32,
            )

            writer.writerow({
                "index": idx,
                "prompt": prompt,
                "response": response,
                "loss": loss,
                "num_loss_tokens": num_tokens,
                "grad_norm": grad_norm,
                "correctness": correctness,
            })

            # Flush frequently so partial progress is saved.
            f.flush()

    print(f"Rank {rank} saved {len(shard_indices)} rows to {output_csv}")


def draw_figs():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    # -----------------------------
    # Settings
    # -----------------------------
    CSV_PATH = "response_grad_norms.rank0.csv"
    # CSV_PATH = "response_grad_norms.with_sensitivity_correctness_corr_0p08.csv"
    GROUP_SIZE = 8

    # Use 0 for population std:
    #   std = sqrt(mean((x - mean)^2))
    #
    # Use 1 if your GRPO implementation uses sample std / torch default correction=1:
    #   std = sqrt(sum((x - mean)^2) / (n - 1))
    STD_DDOF = 0

    EPS = 1e-8

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(CSV_PATH)
    df['grad_norm'] = df['grad_norm']**2

    # Make sure rows are in rollout/group order
    df = df.sort_values("index",ascending=False).reset_index(drop=True)

    # Keep only complete groups of 8
    n_full = (len(df) // GROUP_SIZE) * GROUP_SIZE
    df_full = df.iloc[:n_full].copy()
    df_tail = df.iloc[n_full:].copy()

    print(f"Using {len(df_full)} rows as complete GRPO groups.")
    print(f"Leaving out {len(df_tail)} tail rows.")

    # If you have real group IDs, use those instead.
    # Here we assume every 8 consecutive rows form one GRPO group.
    df_full["group_id"] = np.arange(len(df_full)) // GROUP_SIZE

    # Original correctness is the reward: 0 or 10
    df_full["reward"] = df_full["correctness"].astype(float)

    # -----------------------------
    # Compute legal GRPO advantages
    # -----------------------------
    def compute_grpo_advantage(reward_series):
        rewards = reward_series.to_numpy(dtype=float)
        mean = rewards.mean()
        std = rewards.std(ddof=STD_DDOF)

        # k = 0 or k = 8: all rewards are the same
        # Advantage is conventionally set to 0
        if std < EPS:
            adv = np.zeros_like(rewards)
        else:
            adv = (rewards - mean) / (std + EPS)

        return pd.Series(adv, index=reward_series.index)

    df_full["advantage"] = (
        df_full
        .groupby("group_id")["reward"]
        .transform(compute_grpo_advantage)
    )

    # Track how many correct samples were in each group
    df_full["k_correct_in_group"] = (
        df_full
        .groupby("group_id")["reward"]
        .transform(lambda s: int((s == 10).sum()))
    )

    # -----------------------------
    # Check that advantages are legal
    # -----------------------------
    def legal_advantage_values(group_size=8, ddof=0):
        values = {0.0}

        # sample std scales population-normalized values by sqrt((G - 1) / G)
        scale = 1.0 if ddof == 0 else np.sqrt((group_size - 1) / group_size)

        for k in range(1, group_size):
            pos_adv = scale * np.sqrt((group_size - k) / k)
            neg_adv = -scale * np.sqrt(k / (group_size - k))
            values.add(pos_adv)
            values.add(neg_adv)

        return np.array(sorted(values))

    legal_values = legal_advantage_values(GROUP_SIZE, STD_DDOF)

    observed = set(np.round(df_full["advantage"], 6))
    legal = set(np.round(legal_values, 6))

    illegal = observed - legal
    assert len(illegal) == 0, f"Found illegal advantage values: {illegal}"

    print("Legal advantage values:")
    print(np.round(legal_values, 6))

    # -----------------------------
    # Correlation with grad_norm
    # -----------------------------
    pearson_r, pearson_p = stats.pearsonr(
        df_full["advantage"],
        df_full["grad_norm"]
    )

    spearman_r, spearman_p = stats.spearmanr(
        df_full["advantage"],
        df_full["grad_norm"]
    )

    print(f"Pearson r  = {pearson_r:.3f}, p = {pearson_p:.2e}")
    print(f"Spearman r = {spearman_r:.3f}, p = {spearman_p:.2e}")

    # Under the consecutive-group assumption on your CSV,
    # this should be weakly negative, around:
    # Pearson r  ≈ -0.086
    # Spearman r ≈ -0.095

    # -----------------------------
    # Save transformed CSV
    # -----------------------------
    # df_full.to_csv("response_grad_norms_with_legal_grpo_advantages.csv", index=False)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8,6))

    # Jitter the x-axis slightly because legal advantages are discrete
    rng = np.random.default_rng(0)
    x_jittered = df_full["advantage"] + rng.normal(
        loc=0,
        scale=0.035,
        size=len(df_full)
    )

    ax.scatter(
        x_jittered,
        df_full["grad_norm"],
        s=22,
        alpha=0.35
    )

    # Trend line
    slope, intercept = np.polyfit(
        df_full["advantage"],
        df_full["grad_norm"],
        1
    )

    x_line = np.linspace(
        df_full["advantage"].min(),
        df_full["advantage"].max(),
        200
    )

    y_line = slope * x_line + intercept

    ax.plot(
        x_line,
        y_line,
        linewidth=3,
        color='#67d0d0'
    )

    # Mean grad norm at each legal advantage value
    summary = (
        df_full
        .groupby("advantage")["grad_norm"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    ax.errorbar(
        summary["advantage"][2:-2],
        summary["mean"][2:-2],
        yerr=summary["std"][2:-2] / np.sqrt(summary["count"][2:-2]),
        fmt="D",
        markersize=6,
        capsize=4,
        linewidth=2,
        zorder=5,
        color='#800074'
    )

    # Stats box
    stats_text = (
        f"Pearson r = {pearson_r:.3f}\n"
        f"Spearman r = {spearman_r:.3f}\n"
        f"n = {len(df_full)}\n"
        f"Legal GRPO advantages"
    )

    # ax.text(
    #     0.03,
    #     0.96,
    #     stats_text,
    #     transform=ax.transAxes,
    #     va="top",
    #     ha="left",
    #     fontsize=11,
    #     bbox=dict(
    #         boxstyle="round,pad=0.4",
    #         facecolor="white",
    #         edgecolor="gray",
    #         alpha=0.9
    #     )
    # )

    # ax.set_title(
    #     "Grad norm vs legal GRPO advantage",
    #     fontsize=18,
    #     weight="bold",
    #     pad=16
    # )

    # ax.text(
    #     0.5,
    #     1.01,
    #     "Correctness is converted from reward {0, 10} into group-normalized GRPO advantage.",
    #     transform=ax.transAxes,
    #     ha="center",
    #     fontsize=12,
    #     style="italic"
    # )

    ax.set_xlabel("Advantage", fontsize=22, weight="bold")
    ax.set_ylabel("Gradient Interference", fontsize=22, weight="bold")
    # ax.set_xticks(fontsize=28)
    # ax.set_yticks(fontsize=28)
    ax.set_ylim(bottom=0,top=y_line.max()*2)  # Grad norms are non-negative
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("figs/grad_norm_vs_legal_grpo_advantage.png", dpi=200)
    plt.show()                          


if __name__ == "__main__":
    # main()
    draw_figs()