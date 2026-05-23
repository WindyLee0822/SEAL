# pip install torch transformers matplotlib

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 0) Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "/nobackup2/windy/verl-agent-master/checkpoints/1.5b_82webshop_sft2e-6/global_step_420"
save_name = "sft"
os.makedirs("figs", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()


# -----------------------------
# 1) Choose which parameters to inspect
# -----------------------------
def select_params(m):
    chosen = []

    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        chosen.extend(list(m.transformer.h[-1].parameters()))
        if hasattr(m.transformer, "ln_f"):
            chosen.extend(list(m.transformer.ln_f.parameters()))
    else:
        chosen.extend(list(m.parameters()))

    out = []
    seen = set()
    for p in chosen:
        if p.requires_grad and id(p) not in seen:
            out.append(p)
            seen.add(id(p))
    return out


params = select_params(model)
print("Tracked parameter count:", sum(p.numel() for p in params))


# -----------------------------
# 2) Tokenize prompt + completion
# -----------------------------
def build_example(prompt, completion):
    full_text = prompt + completion

    full = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_only = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    prompt_len = prompt_only["input_ids"].shape[1]

    input_ids = full["input_ids"].to(device)
    attention_mask = full["attention_mask"].to(device)

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_len": prompt_len,
    }


# -----------------------------
# 3) Compute log pi(a|x) and gradient vector
# -----------------------------
def completion_logprob_and_grad(prompt, completion, params):
    batch = build_example(prompt, completion)

    model.zero_grad(set_to_none=True)

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]

    valid = labels != -100
    safe_labels = labels.masked_fill(~valid, 0)

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    logprob = token_log_probs[valid].sum()
    logprob.backward()

    grad_vec = torch.cat([
        (p.grad.detach().float().reshape(-1)
         if p.grad is not None
         else torch.zeros_like(p, dtype=torch.float32).reshape(-1))
        for p in params
    ]).cpu()

    input_ids_cpu = batch["input_ids"][0].detach().cpu()
    label_mask_cpu = (batch["labels"][0].detach().cpu() != -100)

    tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu.tolist())
    completion_tokens = [tok for tok, keep in zip(tokens, label_mask_cpu.tolist()) if keep]

    return {
        "logprob": float(logprob.detach().cpu()),
        "grad": grad_vec,
        "tokens": tokens,
        "completion_tokens": completion_tokens,
    }


# -----------------------------
# 4) Linear algebra helpers
# -----------------------------
def orthonormal_basis(vectors, eps=1e-10):
    basis = []
    for v in vectors:
        r = v.clone().float()
        for q in basis:
            r = r - torch.dot(q, r) * q
        n = r.norm()
        if n > eps:
            basis.append(r / n)

    if len(basis) == 0:
        raise ValueError("No non-zero basis vectors found.")
    return torch.stack(basis, dim=1)  # [P, r]


def fisher_diagnostics_multi(g_pos_list, g_neg_list, eps=1e-10):
    """
    Theory-aligned version:
      - task subspace U = span{delta_i}, delta_i = g_i^+ - g_i^-
      - empirical Fisher F = average over all positive/negative score gradients
      - restricted Fisher F_U = Q^T F Q, where Q spans U
    """
    if len(g_pos_list) != len(g_neg_list):
        raise ValueError("For theory-aligned paired deltas, need same number of positives and negatives.")

    N = len(g_pos_list)
    # g_pos = torch.stack([g.float() for g in g_pos_list], dim=1)   # [P, N]
    # g_neg = torch.stack([g.float() for g in g_neg_list], dim=1)   # [P, N]

    # Comparison directions delta_i = g_i^+ - g_i^-
    # deltas = [g_pos[:, i] - g_neg[:, i] for i in range(N)]
    deltas = torch.cat([g_pos_list,g_neg_list],dim=0)

    # U = span{delta_i}
    # Q = orthonormal_basis(deltas, eps=eps)                        # [P, r]
    # r = Q.shape[1]

    # Empirical Fisher over all 2N gradients
    # G = torch.cat([g_pos, g_neg], dim=1)                          # [P, 2N]
    # FQ = (Q.T @ G @ G.T @ Q) / G.shape[1]                         # [r, r]
    max_vals,min_vals = [],[]
    for t in tqdm(deltas):
        eigvals, eigvecs = [torch.linalg.eigh(t)]
        eigvals = torch.clamp(eigvals, min=0.0)
        max_vals.append(eigvals[0].item())
        min_vals.append(eigvals[-1].item())
        # lam_min = float(eigvals[0].item())
        # lam_max = float(eigvals[-1].item())
        # kappa = lam_max / max(lam_min, eps)
    lam_max = max(max_vals)
    lam_min = min(min_vals)
    kappa = lam_max / max(lam_min, eps)
    # Per-pair task curvatures mu_i = delta_hat_i^T F delta_hat_i
    # mus = []
    # u_coords = []
    # for d in deltas:
    #     dn = d.norm()
    #     if dn.item() > eps:
    #         u = d / dn
    #         mu = float((u @ ((G @ G.T / G.shape[1]) @ u)).item())
    #         mus.append(mu)
    #         u_coords.append((Q.T @ u).cpu())

    # mu_mean = float(np.mean(mus)) if mus else 0.0
    # mu_min = float(np.min(mus)) if mus else 0.0
    # mu_max = float(np.max(mus)) if mus else 0.0
    avg_curvature = float(eigvals.mean().item())

    return {
        # "Q": Q.cpu(),                     # basis of U
        # "rank_U": r,
        # "FQ": FQ.cpu(),                   # restricted Fisher on U
        "eigvals": eigvals.cpu(),         # eigenvalues of F|_U
        "eigvecs": eigvecs.cpu(),
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "kappa_U": kappa,
        # "mu_list": mus,
        # "mu_mean": mu_mean,
        # "mu_min": mu_min,
        # "mu_max": mu_max,
        "avg_curvature": avg_curvature,
        # "u_coords": u_coords,             # coordinates of each normalized delta in Q basis
    }


# -----------------------------
# 5) Visualization
# -----------------------------
def plot_fisher_heatmap(FQ, save_path):
    M = FQ.numpy()
    plt.figure(figsize=(5.5, 5.0))
    plt.imshow(M)
    plt.colorbar(label="curvature")
    labels = [f"q{i+1}" for i in range(M.shape[0])]
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.title("Restricted Fisher on U = span{delta_i}")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, f"{M[i, j]:.2e}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_eigenvalues(eigvals, save_path):
    vals = eigvals.numpy()
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), [f"λ{i+1}" for i in range(len(vals))], rotation=45)
    plt.ylabel("eigenvalue")
    plt.title("Eigenvalues of restricted Fisher F_U")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_curvature_along_task_dirs(mu_list, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(mu_list)), mu_list, marker="o")
    plt.xlabel("pair index i")
    plt.ylabel("μ_i = u_i^T F u_i")
    plt.title("Curvature along paired task directions")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_top2_contours(FQ, u_coords=None, save_path="figs/top2_contours.png"):
    """
    Plot only if rank_U >= 2.
    """
    M = FQ.numpy()
    if M.shape[0] < 2:
        return

    M2 = M[:2, :2]
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = M2[0, 0] * X**2 + 2 * M2[0, 1] * X * Y + M2[1, 1] * Y**2

    plt.figure(figsize=(5, 5))
    plt.contour(X, Y, Z, levels=12)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    if u_coords is not None:
        for idx, u in enumerate(u_coords[:5]):  # show a few
            if len(u) >= 2:
                v = u[:2].numpy()
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                    plt.arrow(0, 0, v[0], v[1], head_width=0.10, length_includes_head=True, alpha=0.6)
                    plt.text(v[0], v[1], f"u{idx}", fontsize=8)

    plt.xlabel("q1")
    plt.ylabel("q2")
    plt.title("Top-2 restricted curvature contours")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------
# 6) Load 32 positive and 32 negative samples
#    Assumption: index-wise pairing is meaningful.
#    If not, replace the pairing rule below.
# -----------------------------
data = json.load(open("draw_webshop.json"))

# Example assumption from your old code:
# item = (prompt, completion, label, ...)
pos_data = [(p, c) for (p, c, lab, *_) in data if lab != 0][:2]
neg_data = [(p, c) for (p, c, lab, *_) in data if lab == 0][:2]

# assert len(pos_data) == 8, f"Need positives, got {len(pos_data)}"
# assert len(neg_data) == 8, f"Need negatives, got {len(neg_data)}"

# -----------------------------
# 7) Compute gradients
# -----------------------------
pos_results = []
neg_results = []
from tqdm import tqdm
for i, (prompt, completion) in tqdm(enumerate(pos_data)):
    # print(f"[pos {i+1}")
    pos_results.append(completion_logprob_and_grad(prompt, completion, params))

for i, (prompt, completion) in tqdm(enumerate(neg_data)):
    # print(f"[neg {i+1}")
    neg_results.append(completion_logprob_and_grad(prompt, completion, params))

g_pos_list = [r["grad"] for r in pos_results]
g_neg_list = [r["grad"] for r in neg_results]

diag = fisher_diagnostics_multi(g_pos_list, g_neg_list)

# -----------------------------
# 8) Print results
# -----------------------------
print("\n=== Aggregate log-probs ===")
print("mean log π(a+|x):", float(np.mean([r["logprob"] for r in pos_results])))
print("mean log π(a-|x):", float(np.mean([r["logprob"] for r in neg_results])))

print("\n=== Aggregate gradient norms ===")
print("mean ||g+||:", float(np.mean([r["grad"].norm().item() for r in pos_results])))
print("mean ||g-||:", float(np.mean([r["grad"].norm().item() for r in neg_results])))
print("mean ||delta_i||:", float(np.mean([(gp - gn).norm().item() for gp, gn in zip(g_pos_list, g_neg_list)])))

print("\n=== Geometry on task subspace U = span{g_i+ - g_i-} ===")
print("rank(U):", diag["rank_U"])
print("lambda_min(F_U):", diag["lambda_min"])
print("lambda_max(F_U):", diag["lambda_max"])
print("kappa(F_U):", diag["kappa_U"])
print("mean task curvature mu_i:", diag["mu_mean"])
print("min task curvature mu_i:", diag["mu_min"])
print("max task curvature mu_i:", diag["mu_max"])
print("average eigenvalue of F_U:", diag["avg_curvature"])
print("Restricted Fisher shape:", tuple(diag["FQ"].shape))
print("Restricted eigenvalues:", diag["eigvals"].numpy())

# -----------------------------
# 9) Visualize
# -----------------------------
plot_fisher_heatmap(diag["FQ"], f"figs/{save_name}_fisher_heatmap.png")
plot_eigenvalues(diag["eigvals"], f"figs/{save_name}_eigenvalues.png")
plot_curvature_along_task_dirs(diag["mu_list"], f"figs/{save_name}_task_curvatures.png")
plot_top2_contours(diag["FQ"], diag["u_coords"], f"figs/{save_name}_top2_contours.png")