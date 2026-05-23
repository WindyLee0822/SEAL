
import json
import re
from collections import Counter
from pathlib import Path
import pandas as pd

def load_outputs(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [row[1] for row in data]

def extract_section(text: str, tag: str) -> str:
    m = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.S | re.I)
    return m.group(1).strip() if m else ""

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    return re.findall(r"\w+|[^\w\s]", s.lower())

def split_clauses(s: str):
    parts = re.split(r"(?<=[.!?])\s+|,\s+|;\s+", s)
    return [p.strip() for p in parts if p.strip()]

def split_sentences(s: str):
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    return [p.strip() for p in parts if p.strip()]

def repeated_ngram_stats(tokens, n=5):
    if len(tokens) < n:
        return 1, 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    max_rep = max(counts.values())
    repeated_mass = sum(v for v in counts.values() if v >= 2)
    return max_rep, repeated_mass / len(ngrams)

def long_span_repeat_stats(tokens, n=10):
    if len(tokens) < n:
        return 1, 0.0
    spans = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(spans)
    max_rep = max(counts.values())
    repeated_mass = sum(v for v in counts.values() if v >= 2)
    return max_rep, repeated_mass / len(spans)

def clause_repeat_ratio(clauses, min_words=6):
    norm = []
    for c in clauses:
        nc = re.sub(r"[^a-z0-9\s]+", " ", c.lower())
        nc = re.sub(r"\s+", " ", nc).strip()
        if len(nc.split()) >= min_words:
            norm.append(nc)
    if not norm:
        return 0.0, 1
    counts = Counter(norm)
    repeated = sum(v for v in counts.values() if v >= 2)
    max_rep = max(counts.values())
    return repeated / len(norm), max_rep

def sentence_dup_rate(sentences, min_chars=20):
    norm = []
    for s in sentences:
        ns = normalize_text(s)
        if len(ns) >= min_chars:
            norm.append(ns)
    if not norm:
        return 0.0, 1
    counts = Counter(norm)
    repeated = sum(v for v in counts.values() if v >= 2)
    max_rep = max(counts.values())
    return repeated / len(norm), max_rep

def format_errors(text: str):
    errs = []
    open_think = len(re.findall(r"<think>", text, flags=re.I))
    close_think = len(re.findall(r"</think>", text, flags=re.I))
    open_answer = len(re.findall(r"<answer>", text, flags=re.I))
    close_answer = len(re.findall(r"</answer>", text, flags=re.I))

    if open_think != 1 or close_think != 1:
        errs.append("think_tag_count")
    if open_answer != 1 or close_answer != 1:
        errs.append("answer_tag_count")

    spaced_close = (
        re.search(r"<\s+/\s*think\s*>", text, flags=re.I)
        or re.search(r"<\s+/\s*answer\s*>", text, flags=re.I)
        or re.search(r"</\s+think\s*>", text, flags=re.I)
        or re.search(r"</\s+answer\s*>", text, flags=re.I)
    )
    if spaced_close:
        errs.append("spaced_closing_tag")

    low = text.lower()
    try:
        ot = low.index("<think>")
        ct = low.index("</think>")
        oa = low.index("<answer>")
        ca = low.index("</answer>")
        if not (ot < ct < oa < ca):
            errs.append("tag_order")
    except ValueError:
        errs.append("missing_tag")

    if text.count("<") != text.count(">"):
        errs.append("unbalanced_angle_brackets")

    return errs

def answer_occurs_in_think(think: str, answer: str) -> int:
    think_n = normalize_text(think)
    ans_n = normalize_text(answer)
    if not ans_n:
        return 0
    return think_n.count(ans_n)

def discourse_repeat_count(think: str) -> int:
    phrases = ["therefore", "so", "thus", "hence", "but the", "however"]
    tn = normalize_text(think)
    return sum(tn.count(p) for p in phrases)

def metrics_for_output(text: str):
    think = extract_section(text, "think")
    answer = extract_section(text, "answer")
    base = think if think else text

    tokens = tokenize(base)
    clauses = split_clauses(base)
    sentences = split_sentences(base)

    max5, rep5 = repeated_ngram_stats(tokens, 5)
    max10, rep10 = long_span_repeat_stats(tokens, 10)
    clause_ratio, max_clause = clause_repeat_ratio(clauses)
    sent_dup, max_sent = sentence_dup_rate(sentences)
    errs = format_errors(text)

    row = {
        "think_tokens": len(tokenize(think)),
        "unique_token_ratio": len(set(tokens)) / len(tokens) if tokens else 1.0,
        "max_5gram_repeat": max5,
        "repeated_5gram_ratio": rep5,
        "max_10gram_repeat": max10,
        "repeated_10gram_ratio": rep10,
        "clause_repeat_ratio": clause_ratio,
        "max_clause_repeat": max_clause,
        "sentence_dup_rate": sent_dup,
        "max_sentence_repeat": max_sent,
        "format_error": int(len(errs) > 0),
        "spaced_closing_tag": int("spaced_closing_tag" in errs),
        "tag_count_error": int("think_tag_count" in errs or "answer_tag_count" in errs),
        "tag_order_error": int("tag_order" in errs),
        "answer_mentions_in_think": answer_occurs_in_think(think, answer),
        "discourse_repeat_count": discourse_repeat_count(think),
    }

    row["local_loop_flag"] = int(
        ((row["max_clause_repeat"] >= 3) and (row["clause_repeat_ratio"] >= 0.25))
        or (row["max_5gram_repeat"] >= 12)
        or ((row["unique_token_ratio"] < 0.18) and (row["think_tokens"] > 40))
    )
    row["sentence_dup_ge_25_flag"] = int(row["sentence_dup_rate"] >= 0.25)
    row["ramble_flag"] = int(
        (row["think_tokens"] >= 180)
        or (row["answer_mentions_in_think"] >= 2)
        or (row["discourse_repeat_count"] >= 6)
    )
    return row

def build_df(path: Path, step_name: str):
    outputs = load_outputs(path)
    df = pd.DataFrame([metrics_for_output(o) for o in outputs])
    df["step"] = step_name
    return df

def summarize(df: pd.DataFrame):
    out = (
        df.groupby("step")
        .agg(
            n_outputs=("step", "size"),
            avg_think_tokens=("think_tokens", "mean"),
            avg_unique_token_ratio=("unique_token_ratio", "mean"),
            avg_max_5gram_repeat=("max_5gram_repeat", "mean"),
            avg_repeated_5gram_ratio=("repeated_5gram_ratio", "mean"),
            avg_clause_repeat_ratio=("clause_repeat_ratio", "mean"),
            local_loop_rate=("local_loop_flag", "mean"),
            avg_sentence_dup_rate=("sentence_dup_rate", "mean"),
            sentence_dup_ge_25_rate=("sentence_dup_ge_25_flag", "mean"),
            format_error_rate=("format_error", "mean"),
            spaced_closing_tag_rate=("spaced_closing_tag", "mean"),
            tag_count_error_rate=("tag_count_error", "mean"),
            avg_answer_mentions_in_think=("answer_mentions_in_think", "mean"),
            avg_discourse_repeat_count=("discourse_repeat_count", "mean"),
            ramble_rate=("ramble_flag", "mean"),
        )
        .reset_index()
    )
    for col in [
        "local_loop_rate",
        "sentence_dup_ge_25_rate",
        "format_error_rate",
        "spaced_closing_tag_rate",
        "tag_count_error_rate",
        "ramble_rate",
    ]:
        out[col] = out[col] * 100
    return out

if __name__ == "__main__":
    dfs = []
    for i in range(100,500,100):
        path = Path(f"data/entropy_analyses/qwen7grpo_step{i}_alfworld.json")
        df = build_df(path, f"step{i}")
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    summary = summarize(all_df)
    summary.to_csv("data/qwen7grpoalfworld_toxicity_metrics_summary.csv", index=False)
    print(summary.round(3))
