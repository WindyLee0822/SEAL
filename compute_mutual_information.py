import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader,DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from tqdm import tqdm
import torch.distributed as dist
import os
import json


def init_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


rank, world_size, local_rank = init_distributed()
DEVICE = torch.device(f"cuda:{local_rank}")
DTYPE = torch.float16

# -----------------------
# Configuration
# -----------------------
MODEL_NAMES = [
    '/nobackup2/windy/verl-agent-master/checkpoints/qwen2.5_0.5b_webshop_part1/global_step_45',
    '/nobackup2/windy/verl-agent-master/checkpoints/qwen2.5_0.5b_webshop_part2/global_step_45',
    '/nobackup2/windy/verl-agent-master/checkpoints/qwen2.5_0.5b_webshop_part3/global_step_45',
    '/nobackup2/windy/verl-agent-master/checkpoints/qwen2.5_0.5b_webshop_part4/global_step_45',
]

PARQUET_PATH = "./data/OpenManus-RL/webshop_test_sft.parquet"
MAX_LENGTH = 4096 
BATCH_SIZE = 1   # increase carefully (memory heavy)

# -----------------------
# Load dataset
# -----------------------
dataset = load_dataset("parquet", data_files=PARQUET_PATH)["train"]
dataset = dataset.map(
    lambda _, i: {"idx": i},
    with_indices=True,
)
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False,
    drop_last=False
)

# -----------------------
# Load models & tokenizers
# -----------------------
models = []

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[0])
# tokenizer.pad_token = tokenizer.eos_token

for name in MODEL_NAMES:
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()
    models.append(model)

# Dataloader
BATCH_SIZE = 4
MAX_LENGTH = 4096

# def collate_fn(batch):
#     """
#     batch: list of dataset rows
#     returns padded input_ids + attention_mask
#     """
#     messages_list = [ex["messages"] for ex in batch]
#     indices = torch.tensor([ex["idx"] for ex in batch], dtype=torch.long)

#     input_ids = [
#         tokenizer.apply_chat_template(
#             messages,
#             tokenize=True,
#             add_generation_prompt=False,
#             max_length=MAX_LENGTH,
#             truncation=True,
#         )
#         for messages in messages_list
#     ]

#     input_ids = torch.nn.utils.rnn.pad_sequence(
#         [torch.tensor(ids) for ids in input_ids],
#         batch_first=True,
#         padding_value=tokenizer.pad_token_id,
#     )

#     attention_mask = input_ids.ne(tokenizer.pad_token_id)

#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "idx": indices,
#     }

def collate_fn(batch):
    """
    Returns:
        input_ids
        attention_mask
        loss_mask (1 = compute loss, 0 = ignore)
        idx
    """

    messages_list = [ex["messages"] for ex in batch]
    indices = torch.tensor([ex["idx"] for ex in batch], dtype=torch.long)

    input_id_list = []
    loss_mask_list = []

    for messages in messages_list:

        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=MAX_LENGTH,
            truncation=True,
        )

        prompt_messages = messages[:-1]

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=MAX_LENGTH,
            truncation=True,
        )

        full_ids = torch.tensor(full_ids, dtype=torch.long)
        prompt_len = len(prompt_ids)

        # 3️⃣ Build loss mask
        loss_mask = torch.zeros(len(full_ids), dtype=torch.long)

        # Only compute loss on assistant part
        loss_mask[prompt_len:] = 1

        input_id_list.append(full_ids)
        loss_mask_list.append(loss_mask)

    # 4️⃣ Pad everything
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_id_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    loss_mask = torch.nn.utils.rnn.pad_sequence(
        loss_mask_list,
        batch_first=True,
        padding_value=0,
    )

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "idx": indices,
    }


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    collate_fn=collate_fn,
)


# -----------------------
# Helper functions
# -----------------------
def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: [vocab]
    """
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

def masked_mean(x, mask):
    """
    x:    (...,)
    mask: same shape, 0 or 1
    """
    mask = mask.float()
    return (x * mask).sum(dim=-1,keepdim=True) / mask.sum(dim=-1,keepdim=True)


def masked_std(x, mask):
    mask = mask.float()
    mean = masked_mean(x, mask)
    var = ((x - mean) ** 2 * mask).sum(dim=-1,keepdim=True) / mask.sum(dim=-1,keepdim=True)
    return torch.sqrt(var)

def mutual_information(logits_list: List[torch.Tensor], loss_mask) -> torch.Tensor:
    """
    logits_list: list of [B, T, V]
    returns: [B] MI per sample
    """
    probs = [F.softmax(l, dim=-1) for l in logits_list]
    mean_probs = torch.mean(torch.stack(probs), dim=0)

    H_mean = compute_entropy(mean_probs)           # [B, T]
    H_models = torch.stack(
        [compute_entropy(p) for p in probs], dim=0
    )                                      # [M, B, T]

    mi = H_mean - H_models.mean(dim=0)     # [B, T]
    # print(mean_probs.shape,H_mean.shape,H_models.shape,mi.shape)

    # normalize 
    mi = (mi - masked_mean(mi,loss_mask)) / masked_std(mi,loss_mask)
    mi = torch.where(loss_mask==0,0,mi)
    return mi 

# -----------------------
# Main loop
# -----------------------
local_mi_scores = []
local_base_logits = []
with torch.no_grad():
    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        loss_mask = batch["loss_mask"].to(DEVICE)
        indices = batch["idx"]

        logits_per_model = []

        for model in models:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            # drop last token (no next-token target)
            logits = outputs.logits[:, :-1, :].float()
            logits_per_model.append(logits)
        
        mi_batch = mutual_information(logits_per_model,loss_mask[:,1:])
        # print(mi_batch.shape,indices.shape)
        for ii,(idx,l) in enumerate(zip(indices.tolist(),mi_batch)):
            local_mi_scores.append((idx,l.tolist()[:attention_mask[ii].sum()-1]))

# -----------------------
# Results
# -----------------------
all_results = [None for _ in range(world_size)]
dist.all_gather_object(all_results, local_mi_scores)

# all_logits = [None for _ in range(world_size)]
# dist.all_gather_object(all_logits, local_base_logits)

if rank == 0:
    # flatten
    flat = [pair for sublist in all_results for pair in sublist]

    # remove redundant elements
    flat_new = []
    num_list = []
    for i,j in flat:
        if i not in num_list:
            num_list.append(i)
            flat_new.append((i,j))
    flat = flat_new

    # flat_logits = [pair for sublist in all_logits for pair in sublist]
    print(len(flat),len(set([i for i, _ in flat])))
    # sort by dataset index
    flat.sort(key=lambda x: x[0])
    # flat_logits.sort(key=lambda x: x[0])
    sorted_indices = [i for i, _ in flat]
    sorted_mi = [mi for _, mi in flat]

    all_eles = torch.tensor([s for mi in sorted_mi for s in mi])
    mean = all_eles.mean().item()
    std = all_eles.std().item()
    sorted_mi = [[(s-mean)/std for s in mi] for _, mi in flat]

    with open('./data/OpenManus-RL/webshop_uncertainty_test.pt','w') as f:
        json.dump(sorted_mi,f)

    # torch.save(flat_logits,'./data/OpenManus-RL/base_p.pt')

