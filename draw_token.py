import math
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
CKPT_OLD = "/nobackup2/windy/verl-agent-master/checkpoints/qwen2.5_1.5b_webshop_sft/global_step_480"
CKPT_NEW = "/nobackup2/windy/verl-agent-master/checkpoints/qwen2.5_1.5b_webshop_sft5e-6/global_step_300"

# One text per line JSONL: {"text": "..."}
DATA_PATH = "./data/OpenManus-RL/webshop_test_sft.parquet"

OUT_DIR = Path("./figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 4096
DEVICE_OLD = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_NEW = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# color normalization caps
PROB_DELTA_CAP = 0.20      # abs(delta prob) bigger than this saturates color
ENTROPY_DELTA_CAP = 2.00   # abs(delta entropy) bigger than this saturates color


# -----------------------------
# Helpers
# -----------------------------
def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )

def load_jsonl(path: str) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
    return texts

def rgba_for_delta(delta: float, cap: float) -> str:
    """
    Blue for positive, red for negative.
    Stronger magnitude => stronger alpha.
    """
    if cap <= 0:
        cap = 1.0
    x = max(-1.0, min(1.0, delta / cap))
    alpha = min(0.85, 0.08 + 0.77 * abs(x))
    if x >= 0:
        # blue
        return f"rgba(30, 90, 255, {alpha:.3f})"
    else:
        # red
        return f"rgba(220, 50, 47, {alpha:.3f})"

def token_texts_from_offsets(text: str, offsets: List[List[int]]) -> List[str]:
    toks = []
    for s, e in offsets:
        if s == e:
            toks.append("")
        else:
            toks.append(text[s:e])
    return toks

@torch.no_grad()
def score_text(
    model,
    tokenizer,
    text: str,
    max_length: int
) -> Dict[str, Any]:
    """
    Returns per-position:
      - target token id x_t
      - token string
      - logprob of observed token
      - prob of observed token
      - entropy of next-token distribution at previous position
    Note:
      logits[:, t-1, :] predicts input_ids[:, t]
    """
    enc = tokenizer.apply_chat_template(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        # return_offsets_mapping=True,
        return_dict=True
    )

    enc_user = tokenizer.apply_chat_template(
        [text[0]],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prompt_len = enc_user.shape[-1]

    input_ids = enc["input_ids"].to(model.device)            # [1, T]
    attention_mask = enc["attention_mask"].to(model.device)  # [1, T]
    # offsets = enc["offset_mapping"][0].tolist()              # [T, 2]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0]   # [T, V]

    # Shift for next-token prediction
    shift_logits = logits[:-1, :]          # predicts token 1..T-1
    shift_targets = input_ids[0, 1:]       # actual tokens 1..T-1
    # shift_offsets = offsets[1:]

    logprobs = F.log_softmax(shift_logits.float(), dim=-1)  # [T-1, V]
    probs = logprobs.exp()

    obs_logprob = logprobs.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)  # [T-1]
    obs_prob = obs_logprob.exp()

    entropy = -(probs * logprobs).sum(dim=-1)  # [T-1]

    # token_texts = token_texts_from_offsets(text, shift_offsets)

    return {
        "input_ids": shift_targets.detach().cpu().tolist()[prompt_len-1:],
        "token_texts": tokenizer.tokenize(tokenizer.apply_chat_template(text, tokenize=False))[prompt_len:],
        "obs_logprob": obs_logprob.detach().cpu().tolist()[prompt_len-1:],
        "obs_prob": obs_prob.detach().cpu().tolist()[prompt_len-1:],
        "entropy": entropy.detach().cpu().tolist()[prompt_len-1:],
    }

def compare_scores(old_s: Dict[str, Any], new_s: Dict[str, Any]) -> Dict[str, Any]:
    if old_s["input_ids"] != new_s["input_ids"]:
        raise ValueError(
            "Token alignment mismatch between checkpoints. "
            "Make sure both checkpoints use the exact same tokenizer."
        )

    out = []
    for tok_id, tok_txt, old_lp, new_lp, old_p, new_p, old_h, new_h in zip(
        old_s["input_ids"],
        old_s["token_texts"],
        old_s["obs_logprob"],
        new_s["obs_logprob"],
        old_s["obs_prob"],
        new_s["obs_prob"],
        old_s["entropy"],
        new_s["entropy"],
    ):
        out.append({
            "token_id": tok_id,
            "token_text": tok_txt,
            "old_logprob": old_lp,
            "new_logprob": new_lp,
            "delta_logprob": new_lp - old_lp,
            "old_prob": old_p,
            "new_prob": new_p,
            "delta_prob": new_p - old_p,
            "old_entropy": old_h,
            "new_entropy": new_h,
            "delta_entropy": new_h - old_h,
        })
    return {"tokens": out}

def render_token_spans(
    compared: Dict[str, Any],
    metric: str = "delta_prob",
    cap: float = 0.20,
    title: str = ""
) -> str:
    spans = []
    for row in compared["tokens"]:
        tok = row["token_text"]
        delta = row[metric]

        # Make whitespace visible/preservable in HTML
        visible_tok = tok.replace("\n", "↵\n").replace("\t", "⇥")
        safe_tok = html_escape(visible_tok) if visible_tok else "∅"

        bg = rgba_for_delta(delta, cap)
        tooltip = (
            f"token={repr(tok)} | "
            f"Δp={row['delta_prob']:+.6f} | "
            f"p_old={row['old_prob']:.6f} | p_new={row['new_prob']:.6f} | "
            f"ΔH={row['delta_entropy']:+.6f} | "
            f"H_old={row['old_entropy']:.6f} | H_new={row['new_entropy']:.6f}"
        )

        spans.append(
            f'<span class="tok" title="{html_escape(tooltip)}" '
            f'style="background:{bg};">{safe_tok}</span>'
        )

    return f"""
    <div class="sample">
      <div class="sample-title">{html_escape(title)}</div>
      <div class="tokens">{''.join(spans)}</div>
    </div>
    """

def write_html(samples_html_prob: List[str], samples_html_entropy: List[str], out_path: Path):
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Qwen checkpoint comparison</title>
<style>
  body {{
    font-family: sans-serif;
    margin: 24px;
    line-height: 1.65;
  }}
  .legend {{
    margin-bottom: 20px;
    padding: 12px 14px;
    border: 1px solid #ddd;
    border-radius: 10px;
    background: #fafafa;
  }}
  .chip {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    color: white;
    margin-right: 8px;
    font-size: 13px;
  }}
  .blue {{ background: rgba(30, 90, 255, 0.85); }}
  .red {{ background: rgba(220, 50, 47, 0.85); }}
  .section-title {{
    margin-top: 28px;
    margin-bottom: 12px;
    font-size: 20px;
    font-weight: 700;
  }}
  .sample {{
    margin-bottom: 24px;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    padding: 14px;
  }}
  .sample-title {{
    font-weight: 700;
    margin-bottom: 10px;
  }}
  .tokens {{
    white-space: pre-wrap;
    word-break: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 15px;
  }}
  .tok {{
    border-radius: 3px;
    padding: 1px 0;
  }}
</style>
</head>
<body>
  <h1>Qwen checkpoint comparison</h1>

  <div class="legend">
    <div><span class="chip blue">Blue</span> increase</div>
    <div style="margin-top:6px;"><span class="chip red">Red</span> decrease</div>
    <div style="margin-top:8px; color:#444;">
      Hover a token to see exact Δp, old/new p, ΔH, and old/new entropy.
    </div>
  </div>

  <div class="section-title">Observed-token probability change</div>
  {''.join(samples_html_prob)}

  <div class="section-title">Entropy change</div>
  {''.join(samples_html_entropy)}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(CKPT_OLD, use_fast=True)

    # Important: both checkpoints should share the same tokenizer/vocab.
    tokenizer_new = AutoTokenizer.from_pretrained(CKPT_NEW, use_fast=True)
    if tokenizer.get_vocab() != tokenizer_new.get_vocab():
        raise ValueError("The two checkpoints do not have identical tokenizer vocabs.")

    model_old = AutoModelForCausalLM.from_pretrained(
        CKPT_OLD,
        torch_dtype=DTYPE,
        device_map=DEVICE_OLD,
    ).eval()

    model_new = AutoModelForCausalLM.from_pretrained(
        CKPT_NEW,
        torch_dtype=DTYPE,
        device_map=DEVICE_NEW,
    ).eval()

    # texts = load_jsonl(DATA_PATH)
    texts = pd.read_parquet(DATA_PATH)


    all_rows = []
    html_prob = []
    html_entropy = []

    for i, text in enumerate(tqdm(texts['messages'].iloc[:32], desc="Scoring texts")):
        old_s = score_text(model_old, tokenizer, text, MAX_LENGTH)
        new_s = score_text(model_new, tokenizer, text, MAX_LENGTH)
        comp = compare_scores(old_s, new_s)

        # Save flat rows too
        for pos, row in enumerate(comp["tokens"]):
            row2 = dict(row)
            row2["sample_idx"] = i
            row2["position"] = pos
            all_rows.append(row2)

        title = f"sample {i}"
        html_prob.append(
            render_token_spans(
                comp,
                metric="delta_prob",
                cap=PROB_DELTA_CAP,
                title=title,
            )
        )
        html_entropy.append(
            render_token_spans(
                comp,
                metric="delta_entropy",
                cap=ENTROPY_DELTA_CAP,
                title=title,
            )
        )

    # JSONL export
    # out_jsonl = OUT_DIR / "per_token_comparison.jsonl"
    # with open(out_jsonl, "w", encoding="utf-8") as f:
    #     for row in all_rows:
    #         f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # HTML export
    out_html = OUT_DIR / "token_diff_view.html"
    write_html(html_prob, html_entropy, out_html)

    # Aggregate summary
    def mean(xs):
        return sum(xs) / max(1, len(xs))

    # summary = {
    #     "num_samples": len(texts),
    #     "num_tokens_scored": len(all_rows),
    #     "mean_delta_prob": mean([r["delta_prob"] for r in all_rows]),
    #     "mean_abs_delta_prob": mean([abs(r["delta_prob"]) for r in all_rows]),
    #     "mean_delta_entropy": mean([r["delta_entropy"] for r in all_rows]),
    #     "mean_abs_delta_entropy": mean([abs(r["delta_entropy"]) for r in all_rows]),
    # }

    # (OUT_DIR / "summary.json").write_text(
    #     json.dumps(summary, indent=2, ensure_ascii=False),
    #     encoding="utf-8"
    # )

    # print(f"Saved JSONL to: {out_jsonl}")
    print(f"Saved HTML  to: {out_html}")
    # print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()