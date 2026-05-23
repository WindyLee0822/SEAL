import json
import re
import argparse
from pathlib import Path
from collections import Counter


FIELD_RE = re.compile(
    r"""(?im)
    ^\s*
    (?:[-*]\s*)?
    (?P<field>
        fabric(?:\s+type)?|
        material|
        fit(?:\s+type)?|
        color(?:\s+type)?|
        size(?:\s+type)?|
        price(?:\s+type)?|
        sleeve(?:\s+type)?|
        outsole|
        sole
    )
    \s*[:\-]\s*
    (?P<value>.+?)
    \s*$
    """,
    re.VERBOSE,
)

DETAIL_INTRO_RE = re.compile(
    r"(following are the details|details of the|details of this|product details|details are as follows)",
    re.I,
)

NEGATIVE_RE = re.compile(
    r"(does not match|doesn't match|not match|not specified|not given|not available|"
    r"not listed|cannot verify|need to verify|no .* match|not the exact|not clear)",
    re.I,
)

POSITIVE_OR_PROCEED_RE = re.compile(
    r"(matches? the required|meets? the criteria|within the required criteria|"
    r"fits? the criteria|only action left|click on ['\"]?buy now|proceed|confirm|buy now)",
    re.I,
)

ONLY_RESULT_RE = re.compile(r"\bonly result\b", re.I)
TOTAL_RESULTS_RE = re.compile(r"Total results:\s*(\d+)", re.I)
ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.I | re.S)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_current_observation(prompt: str) -> str:
    """
    Extracts the current observation block from a WebShop prompt.
    """
    m = re.search(
        r"current observation is:\s*(.*?)\nYour admissible actions",
        prompt,
        flags=re.I | re.S,
    )
    return m.group(1) if m else ""


def extract_prior_actions(prompt: str) -> str:
    """
    Extracts prior action text from the prompt.
    Useful because selected options may only appear in action history.
    """
    m = re.search(
        r"Prior to this step.*?corresponding actions you took:\s*(.*?)\nYou are now at step",
        prompt,
        flags=re.I | re.S,
    )
    return m.group(1) if m else ""


def extract_action(response: str) -> str:
    m = ACTION_RE.search(response)
    return normalize(m.group(1)) if m else ""


def extract_structured_facts(response: str):
    """
    Returns field/value pairs from bullet-like product fact lists.
    Example:
      - Fabric type: cotton spandex
      - Fit type: classic
      - Color type: patina green
    """
    facts = []
    for m in FIELD_RE.finditer(response):
        field = normalize(m.group("field"))
        value = normalize(m.group("value"))
        value = value.strip(" .;,'\"")
        facts.append((field, value))
    return facts


def is_fake_structured_product_facts(prompt: str, response: str) -> bool:
    """
    Flags the pattern where the model creates a structured product-fact list
    and asserts product attributes not grounded in the current observation or
    prior selected actions.

    This intentionally does NOT use the user task as evidence, because copying
    the requested constraints into a product detail list is the hallucination.
    """
    facts = extract_structured_facts(response)

    has_detail_template = bool(DETAIL_INTRO_RE.search(response))
    if not has_detail_template and len(facts) < 2:
        return False

    observation = extract_current_observation(prompt)
    prior_actions = extract_prior_actions(prompt)
    evidence = normalize(observation + " " + prior_actions)

    unsupported = []

    for field, value in facts:
        if not value:
            continue

        # Ignore vague values that are not factual claims.
        if value in {"not given", "not specified", "n.a.", "na", "unknown"}:
            continue

        # Price values often appear as ranges, so check loosely.
        value_tokens = re.findall(r"[a-z0-9.]+", value)
        if not value_tokens:
            continue

        # A fact is supported if the exact value phrase appears in observation
        # or prior action history.
        if value not in evidence:
            unsupported.append((field, value))

    return len(facts) >= 2 and len(unsupported) >= 1


def is_contradictory_rambling(prompt: str, response: str) -> bool:
    """
    Flags rambling contradictions such as:
    - says the product does not match / cannot verify,
      then proceeds to buy or says it matches;
    - says "only result" while the observation reports many results.
    """
    obs = extract_current_observation(prompt)
    action = extract_action(response)

    response_norm = normalize(response)
    obs_norm = normalize(obs)

    has_negative = bool(NEGATIVE_RE.search(response))
    has_positive_or_proceed = bool(POSITIVE_OR_PROCEED_RE.search(response))
    buys_anyway = "buy now" in action

    # Negative uncertainty/mismatch followed by positive conclusion or buy.
    if has_negative and (has_positive_or_proceed or buys_anyway):
        return True

    # "Only result" contradicts a multi-result page.
    if ONLY_RESULT_RE.search(response):
        m = TOTAL_RESULTS_RE.search(obs)
        if m and int(m.group(1)) > 1:
            return True

    # Also catch direct local contradiction: "does not match ... matches"
    if re.search(r"(does not|doesn't|not).*?\bmatch", response_norm) and re.search(
        r"\bmatches?\b.*?(criteria|required)", response_norm
    ):
        return True

    return False


def load_records(path: str):
    """
    Expected JSON format:
      [
        [prompt, response, score, id],
        ...
      ]
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    records = []
    for i, row in enumerate(rows):
        if len(row) < 2:
            continue
        records.append(
            {
                "idx": i,
                "prompt": row[0],
                "response": row[1],
                "score": row[2] if len(row) > 2 else None,
                "episode_id": row[3] if len(row) > 3 else None,
            }
        )
    return records


def hallucination_report(path: str, max_examples: int = 5):
    records = load_records(path)
    total = len(records)

    counts = Counter()
    examples = {
        "fake_structured_product_facts": [],
        "contradictory_rambling": [],
        "any_hallucination": [],
    }

    for r in records:
        fake_structured = is_fake_structured_product_facts(r["prompt"], r["response"])
        contradictory = is_contradictory_rambling(r["prompt"], r["response"])
        any_hallucination = fake_structured or contradictory

        if fake_structured:
            counts["fake_structured_product_facts"] += 1
            if len(examples["fake_structured_product_facts"]) < max_examples:
                examples["fake_structured_product_facts"].append(r)

        if contradictory:
            counts["contradictory_rambling"] += 1
            if len(examples["contradictory_rambling"]) < max_examples:
                examples["contradictory_rambling"].append(r)

        if any_hallucination:
            counts["any_hallucination"] += 1
            if len(examples["any_hallucination"]) < max_examples:
                examples["any_hallucination"].append(r)

    def ratio(name):
        return counts[name] / total if total else 0.0

    report = {
        # "file": str(path),
        # "total_outputs": total,
        # "fake_structured_product_facts_count": counts["fake_structured_product_facts"],
        # "fake_structured_product_facts_ratio": ratio("fake_structured_product_facts"),
        # "contradictory_rambling_count": counts["contradictory_rambling"],
        # "contradictory_rambling_ratio": ratio("contradictory_rambling"),
        # "combined_hallucination_count": counts["any_hallucination"],
        "combined_hallucination_ratio": ratio("any_hallucination"),
    }

    return report, examples


def print_report(report, examples):
    print("\n=== Hallucination ratio report ===")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n=== Example flagged outputs ===")
    for label, rows in examples.items():
        print(f"\n--- {label} ---")
        for r in rows:
            action = extract_action(r["response"])
            snippet = re.sub(r"\s+", " ", r["response"]).strip()[:500]
            print(f"\nidx={r['idx']} episode_id={r['episode_id']} action={action}")
            print(snippet)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("json_files", nargs="+")
    # parser.add_argument("--max-examples", type=int, default=3)
    # args = parser.parse_args()

    for i in [80,160,240,320,400,480,520]:
        path = Path(f"data/entropy_analyses/qwen7grpo_step{i}_webshop.json")
        report, examples = hallucination_report(path, max_examples=0)
        print_report(report, examples)