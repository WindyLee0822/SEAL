# sample_with_vllm.py

import json
from vllm import LLM, SamplingParams
import pandas as pd
from transformers import AutoTokenizer


def sample_responses(
    model_name: str,
    prompts: list,
    output_file: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 256,
    n: int = 3,
):
    """
    Generate sampled responses using vLLM and save to JSONL file.
    """

    # Initialize model
    llm = LLM(model=model_name,gpu_memory_utilization=0.8)

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,  # number of samples per prompt
    )

    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        for idx,output in enumerate(outputs):
            prompt = output.prompt

            for i, candidate in enumerate(output.outputs):
                record = {
                    "prompt": prompt,
                    "prompt_id": idx,
                    "response_id": i,
                    "text": candidate.text,
                    "finish_reason": candidate.finish_reason,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved results to {output_file}")



if __name__ == "__main__":

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # process prompts
    data = pd.read_parquet('./data/OpenManus-RL/test_sft.parquet')
    prompts = []
    for d in data['messages']:
        assert len(d)==2
        # d[0]['content'] += ('\nThis is an example for a response to the question:\n'+ d[1]['content'] + '\nNow answer with a response of your own, including the thinking process:')
        prompts.append(tokenizer.apply_chat_template([d[0]],tokenize=False,add_generation_prompt=True))
    

    prompts = prompts

    sample_responses(
        model_name=model_name,
        prompts=prompts,
        output_file="./data/vllm_sample/alfworld_negtest.jsonl",
        temperature=1,
        top_p=1,
        max_tokens=1024,
        n=8,
    )