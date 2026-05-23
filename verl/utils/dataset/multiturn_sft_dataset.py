# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
import json
import re

class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")

        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]

        # First, get the full conversation tokens
        full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)
        input_ids = full_tokens[0]  # The output is already a tensor
        attention_mask = torch.ones_like(input_ids)

        # Create loss mask by identifying assistant responses
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

        # Process each message to find assistant responses
        for i, msg in enumerate(messages):
            # Get tokens for messages up to this point to find the start position
            prefix_messages = messages[: i + 1]
            prefix_tokens = tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)

            # Get tokens for messages up to previous point
            prev_tokens = tokenizer.apply_chat_template(messages[:i], tokenize=True, return_tensors="pt", add_generation_prompt=False) if i > 0 else None

            # Calculate start and end positions
            start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
            end_pos = prefix_tokens[0].shape[0]

            # If this is an assistant message, set loss mask
            if msg["role"] == "assistant":
                loss_mask[start_pos:end_pos] = 1

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


from copy import deepcopy
class MultiTurnSFTDatasetPost(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.special_tokens = ["<think>","</think>","<action>","</action>"]
        self.action_format="<action>(.*?)</action>"
        self.special_tokens_ids = [id for special_token in self.special_tokens for id in self.tokenizer(special_token,add_special_tokens=False)]
        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

    def __len__(self):
        return len(self.messages)

    def get_se_index(self,messages,message_idx,tokenizer,target):
        messages[message_idx]['content'] = messages[message_idx]['content'].split(target)[0]
        enc = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)[0]
        messages[message_idx]['content'] = messages[message_idx]['content'] + target
        enc1 = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)[0]
        return len(enc[:-2]),len(enc1[:-2]),enc1[len(enc[:-2]):-2]

    def extract_action(self,messages,message_idx,tokenizer):
        match = re.search(self.action_format,messages[message_idx]['content'])
        assert match!=None, (messages[message_idx]['content'])
        action = match.group(1)
        return self.get_se_index(messages,message_idx,tokenizer,action)
    
    def extract_valid_actions_from_webshop(self,messages,tokenizer):
        match = re.search("Your admissible actions of the current situation are: \n(.*?).\n\nNow it's your turn to take one action for the current step.",messages[-2]['content'],re.DOTALL)
        assert match!=None, (messages[-2]['content'])
        valid_actions = match.group(1)
        valid_actions = eval(valid_actions)
        return set([id for action in valid_actions for id in tokenizer(action,add_special_tokens=False)])

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]

        # First, get the full conversation tokens
        full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)
        input_ids = full_tokens[0]  # The output is already a tensor
        attention_mask = torch.ones_like(input_ids)

        # Create loss mask by identifying assistant responses
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        # Process each message to find assistant responses
        # for i, msg in enumerate(messages):
        # Get tokens for messages up to this point to find the start position
        prefix_messages = messages
        prefix_tokens = tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)
        prev_tokens = tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_tensors="pt", add_generation_prompt=True)
        # Calculate start and end positions
        start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
        end_pos = prefix_tokens[0].shape[0]
        loss_mask[start_pos:end_pos] = 1

        for special_token in self.special_tokens:
            s,e,tokens = self.get_se_index(deepcopy(messages),-1,tokenizer,special_token)
            # print(s,e,len(tokens),len(prefix_tokens))
            assert (input_ids[s+1:e-1] == tokens[1:-1]).all(), (input_ids[s:e],tokens)
            loss_mask[s:e] = 2
        s,e,tokens = self.extract_action(deepcopy(messages),-1,tokenizer)
        assert (input_ids[s+1:e-1] == tokens[1:-1]).all(), (tokenizer.decode(input_ids[s:e]),tokenizer.decode(tokens))
        loss_mask[s:e] = 3

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        valid_actions = self.extract_valid_actions_from_webshop(messages,tokenizer)
        padded_action_ids = torch.ones(size=(2048 - sequence_length,), dtype=input_ids.dtype) * pad_token_id
        valid_actions = torch.cat((torch.tensor(list(valid_actions)),padded_action_ids))



        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "special_tokens":self.special_tokens_ids,
            "action_tokens":valid_actions,
        }




# class MultiTurnSFTDatasetPost(Dataset):
#     """
#     Dataset for multi-turn conversations where each assistant response should be trained
#     """

#     def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
#         # Set defaults and extract parameters from config if provided
#         config = config or {}
#         self.truncation = config.get("truncation", "error")
#         self.max_length = config.get("max_length", 1024)
#         # Get messages_key from the new multiturn config structure
#         multiturn_config = config.get("multiturn", {})
#         self.messages_key = multiturn_config.get("messages_key", "messages")

#         assert self.truncation in ["error", "left", "right"]

#         # print(parquet_files[0],parquet_files[1])

#         self.parquet_files = [parquet_files[0]]
#         self.uncertainty_files = [parquet_files[1]]
#         if isinstance(tokenizer, str):
#             tokenizer = hf_tokenizer(tokenizer)
#         self.tokenizer: PreTrainedTokenizer = tokenizer

#         self._download()
#         self._read_files_and_process()

#     def _download(self):
#         for i, parquet_file in enumerate(self.parquet_files):
#             self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)
#         for i, uncertainty_file in enumerate(self.uncertainty_files):
#             self.uncertainty_files[i] = copy_local_path_from_hdfs(uncertainty_file, verbose=True)

#     def _read_files_and_process(self):
#         def series_to_item(ls):
#             import numpy
#             import pandas

#             while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
#                 ls = ls[0]
#             return ls

#         dataframes = []
#         for parquet_file in self.parquet_files:
#             dataframe = pd.read_parquet(parquet_file)
#             dataframes.append(dataframe)
#         self.dataframe = pd.concat(dataframes)

#         # Extract messages list from dataframe
#         self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()
#         import json
#         self.mi = []
#         for uncertainty_file in self.uncertainty_files:
#             self.mi.extend(json.load(open(uncertainty_file)))
#         assert len(self.mi)==len(self.messages),(len(self.mi),len(self.messages))



#     def __len__(self):
#         return len(self.messages)

#     def __getitem__(self, item):
#         tokenizer = self.tokenizer
#         messages = self.messages[item]
#         mi = self.mi[item]

#         # First, get the full conversation tokens
#         full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)
#         input_ids = full_tokens[0]  # The output is already a tensor
#         attention_mask = torch.ones_like(input_ids)
#         assert len(mi)==attention_mask.shape[-1]-1,(len(mi),attention_mask.shape[-1],attention_mask.sum(-1),input_ids.ne(tokenizer.pad_token_id).sum(-1),tokenizer.pad_token,tokenizer.eos_token)
#         mi = torch.tensor(mi,device=attention_mask.device)

#         # Create loss mask by identifying assistant responses
#         loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

#         # Process each message to find assistant responses
#         for i, msg in enumerate(messages):
#             # Get tokens for messages up to this point to find the start position
#             prefix_messages = messages[: i + 1]
#             prefix_tokens = tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)

#             # Get tokens for messages up to previous point
#             prev_tokens = tokenizer.apply_chat_template(messages[:i], tokenize=True, return_tensors="pt", add_generation_prompt=False) if i > 0 else None

#             # Calculate start and end positions
#             start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
#             end_pos = prefix_tokens[0].shape[0]

#             # If this is an assistant message, set loss mask
#             if msg["role"] == "assistant":
#                 loss_mask[start_pos:end_pos] = 1

#         # Handle sequence length
#         sequence_length = input_ids.shape[0]
#         if sequence_length < self.max_length:
#             # Pad sequences
#             pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
#             padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * pad_token_id
#             padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
#             padded_mi = torch.zeros(size=(self.max_length - sequence_length,), dtype=mi.dtype)
#             padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

#             input_ids = torch.cat((input_ids, padded_input_ids))
#             attention_mask = torch.cat((attention_mask, padded_attention_mask))
#             mi = torch.cat((mi,padded_mi))
#             loss_mask = torch.cat((loss_mask, padded_loss_mask))
#         elif sequence_length > self.max_length:
#             if self.truncation == "left":
#                 input_ids = input_ids[-self.max_length :]
#                 attention_mask = attention_mask[-self.max_length :]
#                 mi = mi[-self.max_length :]
#                 loss_mask = loss_mask[-self.max_length :]
#             elif self.truncation == "right":
#                 input_ids = input_ids[: self.max_length]
#                 attention_mask = attention_mask[: self.max_length]
#                 mi = mi[: self.max_length]
#                 loss_mask = loss_mask[: self.max_length]
#             elif self.truncation == "error":
#                 raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
#             else:
#                 raise ValueError(f"Unknown truncation method {self.truncation}")

#         # Create position IDs
#         position_ids = torch.arange(len(input_ids), dtype=torch.long)
#         # Zero out position IDs for padding
#         position_ids = position_ids * attention_mask

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "position_ids": position_ids,
#             "mi":mi,
#             "loss_mask": loss_mask,
#         }

# class MultiTurnSFTDatasetPost(Dataset):
#     """
#     Dataset for multi-turn conversations where each assistant response should be trained
#     """

#     def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
#         # Set defaults and extract parameters from config if provided
#         config = config or {}
#         self.truncation = config.get("truncation", "error")
#         self.max_length = config.get("max_length", 512)
#         self.prompt_max_length = config.get("prompt_max_length", 4096)
#         # Get messages_key from the new multiturn config structure
#         multiturn_config = config.get("multiturn", {})
#         self.messages_key = multiturn_config.get("messages_key", "messages")

#         assert self.truncation in ["error", "left", "right"]

#         # print(parquet_files[0],parquet_files[1])

#         self.parquet_files = [parquet_files[0]]
#         self.jsonl_files = [parquet_files[1]]
#         if isinstance(tokenizer, str):
#             tokenizer = hf_tokenizer(tokenizer)
#         self.tokenizer: PreTrainedTokenizer = tokenizer

#         self._download()
#         self._read_files_and_process()

#     def _download(self):
#         for i, parquet_file in enumerate(self.parquet_files):
#             self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)
#         for i, json_file in enumerate(self.jsonl_files):
#             self.jsonl_files[i] = copy_local_path_from_hdfs(json_file, verbose=True)

#     def _read_files_and_process(self):
#         def series_to_item(ls):
#             import numpy
#             import pandas

#             while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
#                 ls = ls[0]
#             return ls

#         dataframes = []
#         for parquet_file in self.parquet_files:
#             dataframe = pd.read_parquet(parquet_file)
#             dataframes.append(dataframe)
#         self.dataframe = pd.concat(dataframes)

#         # Extract messages list from dataframe
#         messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()


#         json_data = []
#         for json_file in self.jsonl_files:
#             with open(json_file) as f:
#                 json_data.extend([json.loads(line) for line in f])
#         self.resp_num_per_prompt = len(set([d['response_id'] for d in json_data]))
        
#         self.messages = messages
#         self.json_data = json_data
#         # for i in range(len(messages)):
#         #     for j in range(i*self.resp_num_per_prompt, (i+1)*self.resp_num_per_prompt):
#         #         resp = json_data[j]['text']
#         #         if '<action>' in resp and '</action>' in resp and '<think>' in resp and '</think>' in resp \
#         #             and len(self.tokenizer.tokenize(resp))<self.max_length:
#         #             self.messages.append(messages[i])
#         #             self.json_data.append(json_data[j])
#         #             break
        

#         # import json
#         # self.mi = []
#         # for uncertainty_file in self.uncertainty_files:
#         #     self.mi.extend(json.load(open(uncertainty_file)))
#         # assert len(self.mi)==len(self.messages),(len(self.mi),len(self.messages))



#     def __len__(self):
#         return len(self.messages)
    
#     def get_response_index(tokenizer,messages):
#         for i, msg in enumerate(messages):
#             # Get tokens for messages up to this point to find the start position
#             prefix_messages = messages[: i + 1]
#             prefix_tokens = tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)

#             # Get tokens for messages up to previous point
#             prev_tokens = tokenizer.apply_chat_template(messages[:i], tokenize=True, return_tensors="pt", add_generation_prompt=False) if i > 0 else None

#             # Calculate start and end positions
#             start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
#             end_pos = prefix_tokens[0].shape[0]

#             # If this is an assistant message, set loss mask
#             if msg["role"] == "assistant":
#                 return start_pos,end_pos
#         raise ValueError
#     def padded_tokens(self,input_ids,attention_mask,sequence_length):
#         pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
#         padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * pad_token_id
#         padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
#         # padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

#         input_ids = torch.cat((input_ids, padded_input_ids))
#         attention_mask = torch.cat((attention_mask, padded_attention_mask))
#         # loss_mask = torch.cat((loss_mask, padded_loss_mask))
#         return input_ids, attention_mask
    
#     def pack_chat_to_fixed(
#         self,
#         tokenizer,
#         messages,
#         prompt_max_length: int,
#         resp_max_length: int,
#         end_is_eos: bool = True,
#     ):
#         """
#         Returns:
#         input_ids:      (L,) where L = prompt_max_length + resp_max_length
#         attention_mask: (L,) 1 for real tokens, 0 for padding
#         position_ids:   (L,) compacted positions over attention_mask==1
#         prompt_len:     int (# real prompt tokens kept after truncation)
#         resp_len:       int (# real response tokens kept after truncation)

#         Assumption:
#         `messages` is a chat list where the last assistant message is the response you want to supervise.
#         """

#         # 1) Prompt-only ids: use add_generation_prompt=True to stop right before assistant output
#         assert messages[-1]['role']=='assistant' and len(messages)>=2
#         prompt_ids = tokenizer.apply_chat_template(
#             messages[:-1],
#             tokenize=True,
#             return_tensors="pt",
#             add_generation_prompt=True,   # prompt only (expects assistant next)
#         )[0]  # (P,)

#         # 2) Full ids (prompt + response already included in `messages`)
#         full_ids = tokenizer.apply_chat_template(
#             messages,
#             tokenize=True,
#             return_tensors="pt",
#             add_generation_prompt=False,  # includes the assistant content already in messages
#         )[0]  # (F,)

#         P = prompt_ids.numel()
#         F = full_ids.numel()
#         if F < P:
#             raise ValueError(f"Full sequence shorter than prompt? F={F}, P={P}")

#         # 3) Extract response as the suffix after the prompt boundary
#         resp_ids = full_ids[P:]  # (R,)

#         # Optional: ensure response ends with EOS (common for training)
#         if end_is_eos and tokenizer.eos_token_id is not None:
#             if resp_ids.numel() == 0 or resp_ids[-1].item() != tokenizer.eos_token_id:
#                 resp_ids = torch.cat([resp_ids, resp_ids.new_tensor([tokenizer.eos_token_id])], dim=0)

#         # 4) Truncate (keep the last part of prompt? usually keep the *last* tokens)
#         # Prompt: typically keep the most recent tokens (right-truncate / keep tail)
#         if prompt_ids.numel() > prompt_max_length:
#             prompt_ids = prompt_ids[-prompt_max_length:]
#         # Response: keep the beginning (left part) up to resp_max_length
#         if resp_ids.numel() > resp_max_length:
#             resp_ids = resp_ids[:resp_max_length]

#         prompt_len = int(prompt_ids.numel())
#         resp_len = int(resp_ids.numel())

#         # 5) Pad to fixed sizes
#         pad_id = tokenizer.pad_token_id
#         if pad_id is None:
#             # Many decoder-only tokenizers don't define pad; eos is commonly used
#             pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

#         # prompt block
#         prompt_pad = prompt_max_length - prompt_len
#         if prompt_pad > 0:
#             prompt_ids = torch.cat([prompt_ids.new_full((prompt_pad,), pad_id) , prompt_ids], dim=0)

#         # response block
#         resp_pad = resp_max_length - resp_len
#         if resp_pad > 0:
#             resp_ids = torch.cat([resp_ids, resp_ids.new_full((resp_pad,), pad_id)], dim=0)
#         loss_mask = torch.zeros_like(resp_ids)
#         loss_mask[:resp_len] = 1

#         # 6) Concatenate into final fixed-length sequence
#         input_ids = torch.cat([prompt_ids, resp_ids], dim=0)  # (prompt_max_length + resp_max_length,)

#         # 7) Attention mask: 1 for real tokens, 0 for pads
#         attention_mask = torch.zeros_like(input_ids)
#         attention_mask[-prompt_len + prompt_max_length: prompt_max_length] = 1
#         attention_mask[prompt_max_length : prompt_max_length + resp_len] = 1

#         # 8) Position ids: compact over attention_mask==1 (works even for "gappy" masks)
#         position_ids = attention_mask.long().cumsum(dim=0) - 1
#         position_ids = position_ids.clamp(min=0)

#         return input_ids, attention_mask, position_ids, prompt_len, resp_len, loss_mask
    
#     def __getitem__(self, item):
#         tokenizer = self.tokenizer
#         messages = self.messages[item]
#         # generation = self.json_data[item*self.resp_num_per_prompt]
#         try:
#             generation = self.json_data[item]
#         except:
#             generation = self.json_data
        
#         teacher_prompt = messages[0]['content'] + '\nGround Truth:\n'+ messages[1]['content'] + '\nRepeat the ground truth.'
#         teacher_resp = messages[1]['content']
#         teacher_message = [{'role':'user','content':teacher_prompt},{'role':'assistant','content':teacher_resp}]

#         student_message = [messages[0],{'role':'assistant','content':teacher_resp}]
        
#         teacher_input_ids, teacher_attention_mask, teacher_position_ids, t_prompt_len, t_resp_len, t_loss_mask = self.pack_chat_to_fixed(
#             tokenizer, teacher_message, self.prompt_max_length, self.max_length
#         )

#         student_input_ids, student_attention_mask, student_position_ids, s_prompt_len, s_resp_len, s_loss_mask = self.pack_chat_to_fixed(
#             tokenizer, student_message, self.prompt_max_length, self.max_length
#         )
#         assert (t_loss_mask == s_loss_mask).all()

#         return {
#             "input_ids": student_input_ids,
#             "attention_mask": student_attention_mask,
#             "position_ids": student_position_ids,
#             "neg_input_ids": teacher_input_ids,
#             "neg_attention_mask": teacher_attention_mask,
#             "neg_position_ids": teacher_position_ids,
#             "loss_mask": s_loss_mask,
#             "neg_loss_mask": t_loss_mask
#         }


#         # First, get the full conversation tokens
#         # teacher_tokens = tokenizer.apply_chat_template(teacher_message, tokenize=True, return_tensors="pt", add_generation_prompt=False)
#         # teacher_input_ids = teacher_tokens[0]  # The output is already a tensor
#         # teacher_attention_mask = torch.ones_like(teacher_input_ids)

#         # student_tokens = tokenizer.apply_chat_template(student_message, tokenize=True, return_tensors="pt", add_generation_prompt=False)
#         # student_input_ids = student_tokens[0]  # The output is already a tensor
#         # student_attention_mask = torch.ones_like(student_input_ids)
        
#         # # Process each message to find assistant responses
#         # teacher_start,teacher_end = self.get_response_index(teacher_message)
#         # student_start,student_end = self.get_response_index(student_message)
#         # assert student_end - student_start == teacher_end - teacher_start
#         # # Handle sequence length
#         # sequence_length = student_input_ids.shape[0]
#         # if sequence_length < self.max_length:
#         #     student_input_ids,student_attention_mask = self.padded_tokens(student_input_ids,student_attention_mask)
#         #     elif sequence_length > self.max_length:
#         #     if self.truncation == "left":
#         #         student_input_ids,student_attention_mask = student_input_ids[-self.max_length :],student_attention_mask[-self.max_length :]
#         #         student_start,student_end = student_start - (self.max_length - sequence_length),student_end - (self.max_length - sequence_length)
#         #     elif self.truncation == "right":
#         #         student_input_ids,student_attention_mask = student_input_ids[-self.max_length :],student_attention_mask[-self.max_length :]
#         #         student_start,student_end = min(student_start,self.max_length), min(student_end,self.max_length)
#         #     elif self.truncation == "error":
#         #         raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
#         #     else:
#         #         raise ValueError(f"Unknown truncation method {self.truncation}")

#         # sequence_length = teacher_input_ids.shape[0]
#         # if sequence_length < self.max_length:
#         #     teacher_input_ids,teacher_attention_mask = self.padded_tokens(teacher_input_ids,teacher_attention_mask)     
#         # elif sequence_length > self.max_length:
#         #     if self.truncation == "left":
#         #         teacher_start,teacher_end = teacher_start - (self.max_length - sequence_length),teacher_end - (self.max_length - sequence_length)
#         #         teacher_input_ids,teacher_attention_mask = teacher_input_ids[-self.max_length :],teacher_attention_mask[-self.max_length :]
#         #     elif self.truncation == "right":
#         #         teacher_input_ids,teacher_attention_mask = teacher_input_ids[-self.max_length :],teacher_attention_mask[-self.max_length :]
#         #         teacher_start,teacher_end = min(teacher_start,self.max_length), min(teacher_end,self.max_length)
#         #     elif self.truncation == "error":
#         #         raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
#         #     else:
#         #         raise ValueError(f"Unknown truncation method {self.truncation}")

#         # # Create position IDs
#         # position_ids = torch.arange(len(input_ids), dtype=torch.long)
#         # # Zero out position IDs for padding
#         # position_ids = position_ids * attention_mask

#         # return {
#         #     "input_ids": student_input_ids,
#         #     "attention_mask": student_attention_mask,
#         #     "teacher_input_ids": teacher_input_ids,
#         #     "teacher_attention_mask": teacher_attention_mask,
#         #     "position_ids": position_ids,
#         # }


# class MultiTurnSFTDatasetPost(Dataset):
#     """
#     Dataset for multi-turn conversations where each assistant response should be trained
#     """

#     def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
#         # Set defaults and extract parameters from config if provided
#         config = config or {}
#         self.truncation = config.get("truncation", "error")
#         self.max_length = config.get("max_length", 512)
#         self.prompt_max_length = config.get("prompt_max_length", 4096)
#         # Get messages_key from the new multiturn config structure
#         multiturn_config = config.get("multiturn", {})
#         self.messages_key = multiturn_config.get("messages_key", "messages")

#         assert self.truncation in ["error", "left", "right"]
#         self.special_tokens = ["<think>","</think>","<action>","</action>"]
#         self.extract_valid_actions = " "

#         # print(parquet_files[0],parquet_files[1])

#         self.parquet_files = [parquet_files[0]]
#         self.jsonl_files = [parquet_files[1]]
#         if isinstance(tokenizer, str):
#             tokenizer = hf_tokenizer(tokenizer)
#         self.tokenizer: PreTrainedTokenizer = tokenizer
#         self._download()
#         self._read_files_and_process()


#     def _download(self):
#         for i, parquet_file in enumerate(self.parquet_files):
#             self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)
#         for i, json_file in enumerate(self.jsonl_files):
#             self.jsonl_files[i] = copy_local_path_from_hdfs(json_file, verbose=True)

#     def _read_files_and_process(self):
#         def series_to_item(ls):
#             import numpy
#             import pandas

#             while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
#                 ls = ls[0]
#             return ls

#         dataframes = []
#         for parquet_file in self.parquet_files:
#             dataframe = pd.read_parquet(parquet_file)
#             dataframes.append(dataframe)
#         self.dataframe = pd.concat(dataframes)

#         # Extract messages list from dataframe
#         messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()


#         json_data = []
#         for json_file in self.jsonl_files:
#             with open(json_file) as f:
#                 json_data.extend([json.loads(line) for line in f])
#         self.resp_num_per_prompt = len(set([d['response_id'] for d in json_data]))
        
#         self.messages = []
#         self.neg_messages = []
#         # json_data = json_data[len(json_data)//2:] + json_data[:len(json_data)//2]
#         # resp_num_per_prompt = len(set([d['response_id'] for d in json_data]))
#         # for i,mes in enumerate(messages):
#         #     for ii in range(1):
#         #         assert json_data[i*resp_num_per_prompt+ii]['prompt_id']!=i
#         #         self.messages.append(mes)
#         #         self.neg_messages.append([mes[0],{'role':'assistant','content':json_data[i*resp_num_per_prompt+ii]['text']}])
#         neg_messages = messages[len(messages)//2:] + messages[:len(messages)//2]
#         for mes,neg_mes in zip(messages,neg_messages):
#             self.messages.append(mes)
#             self.neg_messages.append([mes[0],neg_mes[1]])

      

#     def __len__(self):
#         return len(self.messages)
    
#     def pack_chat_to_fixed(self,tokenizer,messages):
#         full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)
#         input_ids = full_tokens[0]  # The output is already a tensor
#         attention_mask = torch.ones_like(input_ids)

#         # Create loss mask by identifying assistant responses
#         loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

#         # Process each message to find assistant responses
#         for i, msg in enumerate(messages):
#             # Get tokens for messages up to this point to find the start position
#             prefix_messages = messages[: i + 1]
#             prefix_tokens = tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)

#             # Get tokens for messages up to previous point
#             prev_tokens = tokenizer.apply_chat_template(messages[:i], tokenize=True, return_tensors="pt", add_generation_prompt=False) if i > 0 else None

#             # Calculate start and end positions
#             start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
#             end_pos = prefix_tokens[0].shape[0]

#             # If this is an assistant message, set loss mask
#             if msg["role"] == "assistant":
#                 loss_mask[start_pos:end_pos] = 1

#         # Handle sequence length
#         sequence_length = input_ids.shape[0]
#         if sequence_length < self.max_length:
#             # Pad sequences
#             pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
#             padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * pad_token_id
#             padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
#             padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

#             input_ids = torch.cat((input_ids, padded_input_ids))
#             attention_mask = torch.cat((attention_mask, padded_attention_mask))
#             loss_mask = torch.cat((loss_mask, padded_loss_mask))
#         elif sequence_length > self.max_length:
#             if self.truncation == "left":
#                 input_ids = input_ids[-self.max_length :]
#                 attention_mask = attention_mask[-self.max_length :]
#                 loss_mask = loss_mask[-self.max_length :]
#             elif self.truncation == "right":
#                 input_ids = input_ids[: self.max_length]
#                 attention_mask = attention_mask[: self.max_length]
#                 loss_mask = loss_mask[: self.max_length]
#             elif self.truncation == "error":
#                 raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
#             else:
#                 raise ValueError(f"Unknown truncation method {self.truncation}")

#         # Create position IDs
#         position_ids = torch.arange(len(input_ids), dtype=torch.long)
#         # Zero out position IDs for padding
#         position_ids = position_ids * attention_mask

#         return input_ids,attention_mask,position_ids,loss_mask
    
#     def __getitem__(self, item):
#         tokenizer = self.tokenizer
#         pos_messages = self.messages[item]
#         neg_messages = self.neg_messages[item]
        
        
#         teacher_input_ids, teacher_attention_mask, teacher_position_ids,t_loss_mask = self.pack_chat_to_fixed(
#             tokenizer, neg_messages
#         )

#         student_input_ids, student_attention_mask, student_position_ids,s_loss_mask = self.pack_chat_to_fixed(
#             tokenizer, pos_messages
#         )
        
#         return {
#             "input_ids": student_input_ids,
#             "attention_mask": student_attention_mask,
#             "position_ids": student_position_ids,
#             "neg_input_ids": teacher_input_ids,
#             "neg_attention_mask": teacher_attention_mask,
#             "neg_position_ids": teacher_position_ids,
#             "loss_mask": s_loss_mask,
#             "neg_loss_mask": t_loss_mask
#         }


        


