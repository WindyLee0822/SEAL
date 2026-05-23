import os
import json
import math
import pandas as pd
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional,Tuple

import numpy as np
from sympy import Line2D
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden: (B, T, H)
    attention_mask: (B, T)
    returns: (B, H)
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B, T, 1)
    summed = (last_hidden * mask).sum(dim=1)                  # (B, H)
    denom = mask.sum(dim=1).clamp(min=1e-6)                   # (B, 1)
    return summed / denom


def cls_pool_last_hidden(last_hidden: torch.Tensor) -> torch.Tensor:
    """Take token 0 (CLS for BERT-like)."""
    return last_hidden[:, 0, :]  # (B, H)


# -----------------------------
# Minimal dataset wrapper
# -----------------------------
@dataclass
class TextBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_ids: torch.Tensor
    correctness: torch.Tensor
    # optionally keep labels/ids if you want


def collate_text(batch: List[Dict[str, Any]], tokenizer, max_length: int):
    texts = [i[0] + i[1] for i in batch]
    # action_texts = [i[0] + i[1].split('<action>')[0] + '<action>' for i in batch]
    action_texts = [i[0] for i in batch]
    enc = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    action_enc = tokenizer(
        action_texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    # action_ids = action_enc['attention_mask'].sum(-1) - 1
    correctness = torch.tensor([i[2] for i in batch])
    action_ids = torch.where((action_enc['attention_mask']==0) * (enc['attention_mask']==1),1,0)
    assert (action_ids.sum(-1) > 0).all(), f"{max_length}, {action_ids.sum(-1)}, {action_enc['attention_mask'].sum(-1)}, {enc['attention_mask'].sum(-1)}"
    

    return TextBatch(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        action_ids = action_ids,
        correctness = correctness
    )


# -----------------------------
# Academic-style plotting
# -----------------------------
def academic_style():
    # “Academic-ish” defaults without seaborn and without hard-coding colors.
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def _two_class_masks(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (uniq, m0, m1). Enforces exactly 2 unique label values."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if uniq.size != 2:
        raise ValueError(f"labels must have exactly 2 unique values, got {uniq.size}: {uniq}")
    m0 = labels == uniq[0]
    m1 = labels == uniq[1]
    return uniq, m0, m1


def fit_shared_pca_from_npy(
    model_paths: Dict[str, str],
    n_components: int = 2,
    max_points_per_model: int = 50000,
    seed: int = 0,
) -> PCA:
    """
    Fit ONE PCA on a pooled subset of embeddings from multiple models saved as .npy files.
    This gives a shared projection so plots are comparable across models.
    """
    rng = np.random.default_rng(seed)
    pooled = []
    for name, path in model_paths.items():
        X = np.load(path, mmap_mode="r")  # doesn't fully load into RAM
        n = X.shape[0]
        if n > max_points_per_model:
            idx = rng.choice(n, max_points_per_model, replace=False)
            pooled.append(np.asarray(X[idx]))
        else:
            pooled.append(np.asarray(X))
    pooled = np.concatenate(pooled, axis=0)

    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(pooled)
    return pca


def plot_pca_with_marginals(
    X: np.ndarray,
    outpath: str,
    title: str = "Embedding distribution (PCA)",
    labels: Optional[np.ndarray] = None,
    label_names: tuple = ("Class 0", "Class 1"),
    pca: Optional[PCA] = None,
    fit_pca_if_none: bool = True,
):
    """
    X: (N, H) embeddings
    labels: optional (N,) with exactly two categories (e.g., 0/1, False/True, strings).
    pca: optional sklearn PCA object. If provided, we use pca.transform(X) so multiple models
         can be plotted in the same shared projection.
    fit_pca_if_none: if pca is None and this is True, fit PCA on X (old behavior).
                     if pca is None and this is False, raises.

    Produces: PCA scatter + marginals (histograms on top and right).
    """
    academic_style()

    # ---- PCA to 2D (shared if provided) ----
    if pca is None:
        if not fit_pca_if_none:
            raise ValueError("pca is None and fit_pca_if_none=False. Pass a fitted PCA for shared projection.")
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(X)
    else:
        Z = pca.transform(X)

    # Explained variance ratio only exists for fitted PCA; it will if you passed a fitted one.
    evr = getattr(pca, "explained_variance_ratio_", None)

    # Layout: big scatter + top hist + right hist
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1.2), height_ratios=(1.2, 4), wspace=0.05, hspace=0.05
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    if evr is not None and len(evr) >= 2:
        ax_scatter.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
        ax_scatter.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    else:
        ax_scatter.set_xlabel("PC1")
        ax_scatter.set_ylabel("PC2")
    ax_scatter.set_title(title)

    # -----------------------------
    # If no labels: keep dense view (hexbin + colorbar)
    # -----------------------------
    if labels is None:
        hb = ax_scatter.hexbin(Z[:, 0], Z[:, 1], gridsize=50, mincnt=1)
        cb = fig.colorbar(hb, ax=ax_right, fraction=0.9)
        cb.set_label("Count")

        ax_top.hist(Z[:, 0], bins=40)
        ax_top.set_ylabel("Count")
        ax_top.set_xticklabels([])
        ax_top.grid(True, axis="y")

        ax_right.hist(Z[:, 1], bins=40, orientation="horizontal")
        ax_right.set_xlabel("Count")
        ax_right.set_yticklabels([])
        ax_right.grid(True, axis="x")

    # -----------------------------
    # If labels provided: plot two classes with different colors + centroid arrow
    # -----------------------------
    else:
        _, m0, m1 = _two_class_masks(labels)

        # Scatter with two colors
        ax_scatter.scatter(Z[m0, 0], Z[m0, 1], s=10, alpha=0.55, label=label_names[0])
        ax_scatter.scatter(Z[m1, 0], Z[m1, 1], s=10, alpha=0.55, label=label_names[1])

        # Centroids + arrow (helps show class shift even when mixed)
        c0 = Z[m0].mean(axis=0)
        c1 = Z[m1].mean(axis=0)
        ax_scatter.scatter([c0[0], c1[0]], [c0[1], c1[1]], marker="x", s=140)
        ax_scatter.annotate(
            "",
            xy=(c1[0], c1[1]),
            xytext=(c0[0], c0[1]),
            arrowprops=dict(arrowstyle="->", lw=2),
        )

        ax_scatter.legend(frameon=False, loc="best")

        # Marginals: overlay histograms per class
        ax_top.hist(Z[m0, 0], bins=40, alpha=0.55, label=label_names[0])
        ax_top.hist(Z[m1, 0], bins=40, alpha=0.55, label=label_names[1])
        ax_top.set_ylabel("Count")
        ax_top.set_xticklabels([])
        ax_top.grid(True, axis="y")

        ax_right.hist(Z[m0, 1], bins=40, orientation="horizontal", alpha=0.55)
        ax_right.hist(Z[m1, 1], bins=40, orientation="horizontal", alpha=0.55)
        ax_right.set_xlabel("Count")
        ax_right.set_yticklabels([])
        ax_right.grid(True, axis="x")

    # Tighten
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_norm_hist(
    X: np.ndarray,
    outpath: str,
    title: str = "Embedding L2-norm distribution",
    labels: Optional[np.ndarray] = None,
    label_names: tuple = ("Class 0", "Class 1"),
    bins: int = 60,
):
    """
    Two-class version overlays norm histograms. This is useful to see whether one class tends to
    have systematically larger/smaller embedding magnitude (often an implicit confidence/scale signal),
    and to diagnose collapse (very narrow norms) or outliers (heavy tail).
    """
    academic_style()

    norms = np.linalg.norm(X, axis=1)

    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)

    if labels is None:
        ax.hist(norms, bins=bins)
    else:
        _, m0, m1 = _two_class_masks(labels)
        ax.hist(norms[m0], bins=bins, alpha=0.6, label=label_names[0])
        ax.hist(norms[m1], bins=bins, alpha=0.6, label=label_names[1])
        ax.legend(frameon=False)

    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Count")
    ax.set_title(title)

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="/nobackup2/windy/verl-agent-master/checkpoints/1.5b_simponca/global_step_420")
    # parser.add_argument("--model", type=str, default="/nobackup2/windy/verl-agent-master/checkpoints/1.5b_82webshop_sft2e-6/global_step_420")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    # parser.add_argument("--model", type=str, default="/nobackup2/windy/verl-agent-master/checkpoints/verl_agent_webshop/gigpo_simponcas600/global_step_240/hf")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=4608)
    parser.add_argument("--pooling", type=str, choices=["mean", "cls"], default="mean")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)

    # Outputs
    parser.add_argument("--out_dir", type=str, default="figs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision)
    device = accelerator.device

    # -----------------------------
    # Load data
    # -----------------------------
    # if args.jsonl is not None:
    #     data = []
    #     with open(args.jsonl, "r", encoding="utf-8") as f:
    #         for line in f:
    #             if line.strip():
    #                 data.append(json.loads(line))
    # elif args.hf_dataset is not None:
    #     from datasets import load_dataset
    #     ds = load_dataset(args.hf_dataset, split=args.hf_split)
    #     # turn into list of dicts lazily? simplest: keep as HF dataset; DataLoader works fine.
    #     data = ds
    # else:
    #     raise ValueError("Provide --jsonl or --hf_dataset")
    # data = json.load(open('draw_webshop.json'))
    from datasets import load_dataset
    # data0 = load_dataset('ThornZ/Search-R1-SFT')['train']
    data0 = pd.read_parquet('./data/OpenManus-RL/search_test_sft.parquet')
    data1 = load_dataset('HuggingFaceH4/ultrafeedback_binarized', split='test_prefs')
    data1 = data1.shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    data = []
    
    for idx,d in enumerate(data0['messages'][:2048]) :
        if len(tokenizer.apply_chat_template([d[0]],add_generation_prompt=True,tokenize=True))>=args.max_length-128 \
        or len(d[1]['content'])<32 or len(d[0]['content'])<32:
            print(1)
            continue
        if idx%2==0:
            data.append([tokenizer.apply_chat_template([d[0]],add_generation_prompt=True,tokenize=False),
                         d[1]['content'],
                         1])
        else:
            data.append([tokenizer.apply_chat_template([d[0]],add_generation_prompt=True,tokenize=False),
                        d[1]['content'],
                        0])


    for d in data1.select(range(1024)):
        if len(tokenizer.apply_chat_template([d['chosen'][0]],add_generation_prompt=True,tokenize=True))>=args.max_length-128 \
            or len(d['chosen'][1]['content'])<32 or len(tokenizer.apply_chat_template([d['rejected'][0]],add_generation_prompt=True,tokenize=True))>=args.max_length-128 \
            or len(d['rejected'][1]['content'])<32 or len(d['chosen'][0]['content'])<32 or len(d['rejected'][0]['content'])<32:
            print(1)
            continue
        
        data.append([tokenizer.apply_chat_template([d['chosen'][0]],add_generation_prompt=True,tokenize=False),
                     d['chosen'][1]['content'],
                     2])
        data.append([tokenizer.apply_chat_template([d['rejected'][0]],add_generation_prompt=True,tokenize=False),
                     d['rejected'][1]['content'],
                     3])
        
    print(data[0])
    print(data[-1])

    # with open('./data/vllm_sample/webshop_negtest.jsonl') as f:
    #     ll = [json.loads(line) for line in f]
    #     data = []
    #     num = 8
    #     for i in range(len(ll)//num):
    #         cur = ll[i*num+2]
    #         data.append([cur['prompt'],cur['text'],0])
    # data = []
    # posl = pd.read_parquet('./data/OpenManus-RL/webshop_test_sft.parquet')
    # for l in posl['messages']:
    #     data.append([tokenizer.apply_chat_template([l[0]],add_generation_prompt=True,tokenize=False),
    #                  l[1]['content'],
    #                  1]) 
    # for i,l in enumerate(posl['messages']):
    #     data.append([tokenizer.apply_chat_template([l[0]],add_generation_prompt=True,tokenize=False),
    #                  posl['messages'].iloc[-i][1]['content'],
    #                  0])    
        
    
    # print(data[0],data[-1])
    

    # -----------------------------
    # Load tokenizer/model
    # -----------------------------
    
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    # DataLoader
    dl = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_text(b, tokenizer, args.max_length),
        num_workers=4,
        pin_memory=True,
    )

    # Prepare for distributed
    model, dl = accelerator.prepare(model, dl)

    # -----------------------------
    # Extract embeddings
    # -----------------------------
    all_embeds = []
    all_correctness = []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            action_ids = batch.action_ids.to(device)
            correctness = batch.correctness.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]  # (B,T,H)

            # if args.pooling == "mean":
            #     emb = mean_pool_last_hidden(last_hidden, attention_mask)  # (B,H)
            # else:
            #     emb = cls_pool_last_hidden(last_hidden)  # (B,H)
            # emb = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), action_ids]   # (B, H)
            # assert (action_ids[:,1:].sum(dim=-1)!=0).all()
            emb = (last_hidden[:,:-1,:] * action_ids[:,1:].unsqueeze(-1)).sum(dim=1)/action_ids[:,1:].sum(-1)[...,None]
            # emb = (last_hidden * action_ids.unsqueeze(-1)).sum(dim=1)/action_ids.sum(-1)[...,None]
            # Gather across processes so rank0 ends up with the full array
            emb_gathered = accelerator.gather_for_metrics(emb)  # (B_total, H) across ranks
            correct_gathered = accelerator.gather_for_metrics(correctness)
            all_embeds.append(emb_gathered.cpu())
            all_correctness.append(correct_gathered.cpu())

    embeds = torch.cat(all_embeds, dim=0).numpy()  # (N, H) on every rank due to gather_for_metrics behavior
    correctness = torch.cat(all_correctness,dim=0).numpy()
    assert embeds.shape[0] == correctness.shape[0], (embeds.shape, correctness.shape)
    # If you want only rank0 to write files:
    if accelerator.is_main_process:
        npy_path = os.path.join(args.out_dir, "search_embeddings.npy")
        np.save(npy_path, embeds)
        npy_path = os.path.join(args.out_dir, "search_labels.npy")
        np.save(npy_path, correctness)

    accelerator.wait_for_everyone()


def draw_pic():
    model_paths = {
        "webshop": "figs/webshop_embeddings.npy",
        # "sft": "figs/sft_embeddings.npy",
        # "neg": "figs/neg_embeddings.npy"
    }
    labels = np.load("figs/labels.npy")

    save_dir = "figs"
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # Academic plotting style
    # =========================
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    CLASS_COLORS = ["#4C72B0", "#DD8452"]   # blue / orange
    CLASS_NAMES = ["Class 0", "Class 1"]    # replace with your real names if needed

    # =========================
    # Sanity check
    # =========================
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")

    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError(f"Expected 2 classes, got {unique_labels}")

    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

    # =========================
    # t-SNE helper
    # =========================
    def compute_tsne(x, seed=42):
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {x.shape}")
        if x.shape[0] != len(labels):
            raise ValueError(f"Embedding rows ({x.shape[0]}) != label count ({len(labels)})")

        # # Optional PCA before t-SNE for stability
        n_pca = min(50, x.shape[1], x.shape[0] - 1)
        if n_pca >= 2:
            x = PCA(n_components=n_pca, random_state=seed).fit_transform(x)

        perplexity = 100

        z = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=1000,
            random_state=seed,
        ).fit_transform(x)

        # Normalize for nicer display
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)
        return z

    # =========================
    # Generate 3 separate figures
    # =========================
    for model_name, path in model_paths.items():
        emb = np.load(path)
        z = compute_tsne(emb)

        fig, ax = plt.subplots(figsize=(5.2, 4.6))

        for lab in unique_labels:
            mask = labels == lab
            idx = label_to_idx[lab]
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                s=22,
                alpha=0.85,
                c=CLASS_COLORS[idx],
                label=CLASS_NAMES[idx],
                edgecolors="white",
                linewidths=0.35,
            )

        ax.set_title(model_name.upper())
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False, loc="best")

        png_path = os.path.join(save_dir, f"tsne_{model_name}.png")
        # pdf_path = os.path.join(save_dir, f"{model_name}_tsne.pdf")

        plt.savefig(png_path, dpi=400, bbox_inches="tight")
        # plt.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {png_path}")
        # print(f"Saved: {pdf_path}")


def cal_similarity():
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    model_paths = {
        # "pretrain": "figs/pretrain_embeddings.npy",
        "sft": "figs/sft_embeddings.npy",
        "neg": "figs/neg_embeddings.npy",
        "rl": "figs/rl_embeddings.npy"
    }

    labels = np.load("figs/labels.npy")
    labels = np.asarray(labels)

    # sanity check
    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError(f"Expected exactly 2 categories, got {unique_labels}")

    label_a, label_b = unique_labels[0], unique_labels[1]

    def average_interclass_cosine_similarity(embeddings, labels, label_a, label_b):
        """
        Compute average cosine similarity between all points in class A and all points in class B.
        """
        a = embeddings[labels == label_a]
        b = embeddings[labels == label_b]

        if len(a) == 0 or len(b) == 0:
            raise ValueError("One class has no samples.")

        sim_matrix = cosine_similarity(a, b)   # shape: [n_a, n_b]
        return sim_matrix.mean()

    for model_name, path in model_paths.items():
        emb = np.load(path)

        if emb.shape[0] != len(labels):
            raise ValueError(
                f"{model_name}: number of embeddings ({emb.shape[0]}) "
                f"does not match number of labels ({len(labels)})"
            )

        avg_sim = average_interclass_cosine_similarity(emb, labels, label_a, label_b)
        pp_sim = average_interclass_cosine_similarity(emb, labels, label_a, label_a)
        nn_sim = average_interclass_cosine_similarity(emb, labels, label_b, label_b)
        print(f"{model_name},{label_a},{label_b} = {avg_sim:.6f}")
        print(f"{model_name},{label_a},{label_a} = {pp_sim:.6f}")
        print(f"{model_name},{label_b},{label_b} = {nn_sim:.6f}")
        print(f"ratio = {avg_sim*2-(pp_sim+nn_sim):.6f}")
        



def draw_again():
    import os
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"  # helps in some notebook/container environments

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # save figures without opening a GUI
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D


    # Load data
    emb = np.load("figs/search_embeddings.npy")
    labels = np.load("figs/search_labels.npy").reshape(-1)
    write_name = 'search'

    assert emb.shape[0] == labels.shape[0], "Mismatch between embeddings and labels."
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 22,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ---------- 1) PCA to 2D ----------
    emb = np.array([emb[i] for i in range(len(labels)) if labels[i] in [0,1]])
    labels = np.array([labels[i] for i in range(len(labels)) if labels[i] in [0,1]])
    X = emb.astype(float)
    X_centered = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    print(emb,X.mean(axis=0, keepdims=True),X_centered)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_pca = X_centered @ Vt[:2].T

    mask0 = labels == 0
    mask1 = labels == 1
    mask2 = labels == 2
    mask3 = labels == 3


    # ---------- 2) Spread in original embedding space ----------
    centroid0 = X[mask0].mean(axis=0)
    centroid1 = X[mask1].mean(axis=0)
    centroid2 = X[mask2].mean(axis=0)
    centroid3 = X[mask3].mean(axis=0)

    dist0 = np.linalg.norm(X[mask0] - centroid0, axis=1)
    dist1 = np.linalg.norm(X[mask1] - centroid1, axis=1)
    dist2 = np.linalg.norm(X[mask2] - centroid2, axis=1)
    dist3 = np.linalg.norm(X[mask3] - centroid3, axis=1)

    # ---------- 3) Spread circles in PCA space ----------
    centroid0_2d = X_pca[mask0].mean(axis=0)
    centroid1_2d = X_pca[mask1].mean(axis=0)
    centroid2_2d = X_pca[mask2].mean(axis=0)
    centroid3_2d = X_pca[mask3].mean(axis=0)

    r0 = np.mean(np.linalg.norm(X_pca[mask0] - centroid0_2d, axis=1))
    r1 = np.mean(np.linalg.norm(X_pca[mask1] - centroid1_2d, axis=1))
    r2 = np.mean(np.linalg.norm(X_pca[mask2] - centroid2_2d, axis=1))
    r3 = np.mean(np.linalg.norm(X_pca[mask3] - centroid3_2d, axis=1))

    # ---------- 4) Plot PCA scatter ----------
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    ax.scatter(X_pca[mask0, 0], X_pca[mask0, 1], s=10, alpha=0.35, label="agent",color="#800074")
    ax.scatter(X_pca[mask1, 0], X_pca[mask1, 1], s=10, alpha=0.35, label="agent",color="#67d0d0")
    # ax.scatter(X_pca[mask2, 0], X_pca[mask2, 1], s=10, alpha=0.35, label="alignment",color="#800046")
    # ax.scatter(X_pca[mask3, 0], X_pca[mask3, 1], s=10, alpha=0.35, label="alignment",color="#298c8c")

    # ax.scatter([centroid0_2d[0]], [centroid0_2d[1]], marker="x", s=120, linewidths=2, label="centroid 0")
    # ax.scatter([centroid1_2d[0]], [centroid1_2d[1]], marker="x", s=120, linewidths=2, label="centroid 1")

    # ax.add_patch(plt.Circle(tuple(centroid0_2d), r0/1.1, fill=False, linewidth=2, alpha=0.9))
    # ax.add_patch(plt.Circle(tuple(centroid1_2d), r1*1.5, fill=False, linewidth=2, alpha=0.9))

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    scale=1.5
    ax.set_xlim(X_pca[:,0].min()*scale,X_pca[:,0].max()*scale )
    ax.set_ylim(X_pca[:,1].min()*scale,X_pca[:,1].max()*scale )
    # ax.set_title("2D PCA projection of embeddings")
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=8, label=f'{write_name}-p',color='#800074'),
        # Line2D([0], [0], marker='o', linestyle='None', markersize=8, label='alignment-p',color='#800074'),
        Line2D([0], [0], marker='o', linestyle='None', markersize=8, label=f'{write_name}-n',color='#67d0d0'),
        # Line2D([0], [0], marker='o', linestyle='None', markersize=8, label='alignment-n',color='#298c8c'),
        ]
    ax.legend(handles=legend_handles)
    # ax.legend()
    fig.tight_layout()
    fig.savefig(f"figs/tsne_{write_name}1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


    # ---------- 5) Plot boxplot of distances to centroid ----------
    # fig = plt.figure(figsize=(7, 5))
    # ax = fig.add_subplot(111)

    # ax.boxplot([dist0, dist1], tick_labels=["agent", "alignment"], showfliers=False)
    # ax.set_ylabel("Distance to class centroid")
    # ax.set_title("Within-class spread in original embedding space")

    # fig.tight_layout()
    # fig.savefig("figs/tsne_webshop1.png", dpi=180, bbox_inches="tight")
    # plt.close(fig)


    # ---------- 6) Print summary ----------
    print(f"n(label 0) = {mask0.sum()}, n(label 1) = {mask1.sum()}")
    print(f"Mean distance to centroid — label 0: {dist0.mean():.2f}, label 1: {dist1.mean():.2f}")
    print(f"Median distance to centroid — label 0: {np.median(dist0):.2f}, label 1: {np.median(dist1):.2f}")
    print(f"2D PCA mean radius — label 0: {r0:.2f}, label 1: {r1:.2f}")
    print("Saved: embedding_pca_spread.png")
    print("Saved: embedding_centroid_distance_boxplot.png") 

def draw_again_fair(method="pca", seed=42, l2_normalize=True, max_points_per_class=None):
    import os
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)

    # -----------------------------
    # Load data
    # -----------------------------
    emb = np.load("figs/alf_embeddings.npy")
    labels = np.load("figs/alf_labels.npy").reshape(-1)

    assert emb.shape[0] == labels.shape[0], "Mismatch between embeddings and labels."

    X = emb.astype(np.float64)

    # Remove NaN / Inf rows
    good = np.isfinite(X).all(axis=1) & np.isfinite(labels)
    X = X[good]
    labels = labels[good]

    # Optional but usually fairer for embeddings:
    # prevents vector magnitude from dominating the projection.
    if l2_normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # Optional balanced sampling for visualization.
    # This avoids one class visually dominating the plot.
    # If you use this, say in the caption that the plot is class-balanced.
    if max_points_per_class is not None:
        keep = []
        for k in sorted(np.unique(labels)):
            idx = np.where(labels == k)[0]
            if len(idx) > max_points_per_class:
                idx = rng.choice(idx, size=max_points_per_class, replace=False)
            keep.append(idx)
        keep = np.concatenate(keep)
        keep = rng.permutation(keep)
        X = X[keep]
        labels = labels[keep]

    mask0 = labels == 0
    mask1 = labels == 1
    mask2 = labels == 2
    mask3 = labels == 3

    print("Counts:", {
        0: int(mask0.sum()),
        1: int(mask1.sum()),
        2: int(mask2.sum()),
        3: int(mask3.sum()),
    })

    # -----------------------------
    # Fair unsupervised projection
    # -----------------------------
    method = method.lower()

    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        Z = reducer.fit_transform(X)
        title = "PCA projection of embeddings"
        xlabel = f"PC1 ({reducer.explained_variance_ratio_[0] * 100:.1f}% var)"
        ylabel = f"PC2 ({reducer.explained_variance_ratio_[1] * 100:.1f}% var)"
        outpath = "figs/pca_four_class_divergence.png"

    elif method in ["tsne", "t-sne"]:
        # Standard practice: reduce high-dimensional embeddings to 50D with PCA first,
        # then run t-SNE.
        n_pre = min(50, X.shape[1], X.shape[0] - 1)
        X_pre = PCA(n_components=n_pre, random_state=seed).fit_transform(X)

        perplexity = min(30, max(5, (X.shape[0] - 1) // 3))

        try:
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                max_iter=1000,
                random_state=seed,
            )
        except TypeError:
            # For older sklearn versions
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                n_iter=1000,
                random_state=seed,
            )

        Z = reducer.fit_transform(X_pre)
        title = "t-SNE projection of embeddings"
        xlabel = "t-SNE 1"
        ylabel = "t-SNE 2"
        outpath = "figs/tsne_alfworld.png"

    else:
        raise ValueError("method must be 'pca' or 'tsne'.")

    # -----------------------------
    # Quantitative divergence in original embedding space
    # -----------------------------
    unique = sorted(np.unique(labels))
    centroids = {k: X[labels == k].mean(axis=0) for k in unique}
    spreads = {
        k: np.linalg.norm(X[labels == k] - centroids[k], axis=1).mean()
        for k in unique
    }

    print("\nMean within-class spread in original embedding space:")
    for k in unique:
        print(f"label {k}: {spreads[k]:.4f}")

    print("\nPairwise centroid distances in original embedding space:")
    for i, a in enumerate(unique):
        for b in unique[i + 1:]:
            d = np.linalg.norm(centroids[a] - centroids[b])
            print(f"label {a} vs label {b}: {d:.4f}")

    if len(unique) >= 2 and X.shape[0] > len(unique):
        sil = silhouette_score(X, labels, metric="euclidean")
        print(f"\nSilhouette score in original embedding space: {sil:.4f}")

    # -----------------------------
    # Plot helpers
    # -----------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 22,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    colors = {
        0: "#FF09E6",
        1: "#800046",
        2: "#56ffff",
        3: "#298c8c",
    }

    names = {
        0: "alfworld-p",
        1: "alfworld-n",
        2: "alignment-p",
        3: "alignment-n",
    }

    def add_cov_ellipse(ax, Z_class, color, n_std=1.5):
        if Z_class.shape[0] < 3:
            return

        mean = Z_class.mean(axis=0)
        cov = np.cov(Z_class.T)

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))

        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            linewidth=2.0,
            alpha=0.9,
            color=color,
        )
        ax.add_patch(ellipse)

    # -----------------------------
    # Plot
    # -----------------------------
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    for k in unique:
        mk = labels == k
        ax.scatter(
            Z[mk, 0],
            Z[mk, 1],
            s=10,
            alpha=0.35,
            color=colors.get(k, None),
            label=names.get(k, f"label {k}"),
        )

        # Centroid
        # cz = Z[mk].mean(axis=0)
        # ax.scatter(
        #     [cz[0]],
        #     [cz[1]],
        #     marker="x",
        #     s=140,
        #     linewidths=2.5,
        #     color=colors.get(k, None),
        # )

        # Ellipse showing spread/divergence
        # add_cov_ellipse(ax, Z[mk], colors.get(k, "black"), n_std=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8,
            label=names[k],
            color=colors[k],
        )
        for k in unique
    ]

    ax.legend(handles=legend_handles, frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {outpath}")

if __name__ == "__main__":
    # main()
    # cal_similarity()
    draw_again()
    # draw_again_fair(method="pca", l2_normalize=False, max_points_per_class=512)
    # draw_again_fair(method="tsne", l2_normalize=False, max_points_per_class=800)