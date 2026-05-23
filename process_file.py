from datasets import load_dataset, DatasetDict
from pathlib import Path
import pandas as pd
import random
random.seed(0)

ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""




def process_alfworld():
    history_length = 10
    # transform the manus data to verl-agent data
    def split_one_traj_to_multiple_samples(uid, traj):

        def split_thought_and_action(string):
            slist = string.split('Action: ')
            slist = [s.replace('Thought: ','').strip() for s in slist if s.replace('Thought: ','').strip()!='']
            try:
                assert len(slist)==2 if 'Thought: ' in string else len(slist)==1,(string, slist)
            except:
                slist = string.split('\n')
                slist = [s.replace('Thought: ','').strip() for s in slist if s.replace('Thought: ','').strip()!='']
                assert len(slist)<=2

            if len(slist)==2:
                return '<think>\n'+slist[0].strip()+'\n</think>\n\n<action>'+slist[1].strip()+'</action>',slist[1].strip()
            else:
                return '<action>'+slist[0].strip()+'</action>',slist[0].strip()
        
        def form_conversation(prompt,response):
            return [{'role':'user','content':prompt},{'role':'assistant','content':response}]
        
        def form_history(begin_id,observation_list,action_list):
            string = ""
            for i,(obs,action) in enumerate(zip(observation_list,action_list)):
                string += f"[Observation {begin_id+i}: '{obs}', Action {begin_id+i}: '{action}']\n"
            return string.strip()
        
        def form_action_string(admissible_action_list):
            return '\n '.join([f"'{ele}'" for ele in admissible_action_list if ele!=''])

        assert len(traj)>2, traj
        traj = traj[2:]
        assert traj[0]['role']=='user'
        instruction = traj[0]['content'].split('\nYour task is to: ')
        observation_list = ['-= Welcome to TextWorld, ALFRED! =-\n\n' + instruction[0].strip() + '\n\nYour task is to: '+ instruction[1].split('\nAVAILABLE ACTIONS: ')[0].strip()]
        action_list= [split_thought_and_action(traj[1]['content'])[1]]
        response_list = [split_thought_and_action(traj[1]['content'])[0]]
        task_description = instruction[1].split('\nAVAILABLE ACTIONS: ')[0].strip()
        admissible_actions = instruction[1].split('\nAVAILABLE ACTIONS: ')[1].strip().split(',')


        final_data_list = [[uid,form_conversation(ALFWORLD_TEMPLATE_NO_HIS.format(current_observation=observation_list[-1],
                                                                            admissible_actions=form_action_string(admissible_actions)),
                                            response_list[0])]]

        for i, ele in enumerate(traj[2:]):
            if i%2==0:
                assert ele['role']=='user'
                observation_list.append(ele['content'].strip() if 'go to' not in action_list[-1] else f'You arrive at {action_list[-1].replace('go to ','')}. '+ele['content'].strip())
            else:
                assert ele['role']=='assistant'
                action_list.append(split_thought_and_action(ele['content'])[1])
                response_list.append(split_thought_and_action(ele['content'])[0])
                assert i//2 == len(response_list)-2
                assert len(observation_list)==len(action_list)
                final_data_list.append([uid, form_conversation(ALFWORLD_TEMPLATE.format(
                    task_description=task_description,step_count=i//2+1,history_length=min(i//2+1, history_length),action_history=form_history(i//2+1-min(i//2, history_length-1),observation_list[-1-history_length:-1],action_list[-1-history_length:-1]),
                    current_step=i//2+2,current_observation=observation_list[-1],admissible_actions=form_action_string(admissible_actions if action_list[-1] in admissible_actions else admissible_actions+[action_list[-1]])),
                                            response_list[-1])]) 

        return final_data_list 

    # Load the dataset (all splits)
    dataset = load_dataset("CharlieDreemur/OpenManus-RL")

    def filter_trial_ids(example):
        return "_trial_" in example.get("id", "").lower()

    # Apply filtering to every split
    train_trials = dataset["train"].filter(filter_trial_ids)

    output_dir = Path("./data/OpenManus-RL/")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = "./data/OpenManus-RL/train_sft.parquet"

    all_dataset = []
    for idx,data in enumerate(train_trials.select(range(2048))):
        all_dataset.extend(split_one_traj_to_multiple_samples(idx,data['conversations']))

    df = pd.DataFrame(all_dataset, columns=["id", "messages"])
    df.to_parquet("./data/OpenManus-RL/train_sft.parquet", engine="pyarrow", index=False)
    print(f"{len(all_dataset)} examples")

    all_dataset = []
    for idx,data in enumerate(train_trials.select(range(2048,len(train_trials)))):
        all_dataset.extend(split_one_traj_to_multiple_samples(2048+idx,data['conversations']))

    df = pd.DataFrame(all_dataset, columns=["id", "messages"])
    df.to_parquet("./data/OpenManus-RL/test_sft.parquet", engine="pyarrow", index=False)

    # Quick sanity check
    print(f"{len(all_dataset)} examples")
    # Example: inspect one item
    # print(all_dataset[0])
    # print(all_dataset[-2])
    # print(all_dataset[-1])


def process_webshop():
    history_length = 2
    pass_num =0
    # transform the manus data to verl-agent data
    def split_one_traj_to_multiple_samples(uid, traj):
        nonlocal pass_num
        def split_thought_and_action(string):
            slist = string.split('Action:')
            slist = [s.replace('Thought:','').strip() for s in slist if s.replace('Thought: ','').strip()!='']
            try:
                assert len(slist)==2 if 'Thought:' in string else len(slist)==1,(string, slist)
            except:
                slist = string.split('\n')
                slist = [s.replace('Thought:','').strip() for s in slist if s.replace('Thought: ','').strip()!='']
                assert len(slist)<=2, string

            if len(slist)==2:
                return '<think>\n'+slist[0].strip()+'\n</think>\n\n<action>'+slist[1].strip().lower()+'</action>',slist[1].strip()
            else:
                return '<action>'+slist[0].strip().lower()+'</action>',slist[0].strip().lower()
        
        def form_conversation(prompt,response):
            return [{'role':'user','content':prompt},{'role':'assistant','content':response}]
        
        def form_history(begin_id,observation_list,action_list):
            string = ""
            for i,(obs,action) in enumerate(zip(observation_list,action_list)):
                string += f"[Observation {begin_id+i}: '{obs}', Action {begin_id+i}: '{action}']\n"
            return string.strip()
        
        def form_action_string(admissible_action_list):
            return ',\n'.join([f"'{ele}'" for ele in admissible_action_list if ele!=''])
        
        def admissible_action_from_observation(observation,action):
            action_list = []            
            obs_list = observation.split('[SEP]')
            for obs in obs_list:
                obs = obs.strip(" \n'")
                if 'price' in obs.lower() or 'size'==obs.lower() or 'color'==obs.lower() or len(obs)>10:
                    continue
                if 'search' == obs.lower():
                    action_list.append('search[<your query>]')
                action_list.append(f'click[{obs.lower()}]')

            if action not in action_list:
                action_list.append(action)
            return action_list

        
        assert len(traj)>2, traj
        traj = traj[2:]

        
        assert traj[0]['role']=='user'
        instruction = traj[0]['content'].split('[SEP]')
        observation_list = ["'"+instruction[-1].strip()+"'"]
        action_list= [split_thought_and_action(traj[1]['content'])[1]]
        response_list = [split_thought_and_action(traj[1]['content'])[0]]
        task_description = instruction[-2].strip()
        admissible_actions = admissible_action_from_observation(observation_list[-1],action_list[-1])


        final_data_list = [[uid,form_conversation(WEBSHOP_TEMPLATE_NO_HIS.format(task_description =task_description,
                                                                            current_observation=observation_list[-1],
                                                                            available_actions=form_action_string(admissible_actions)),
                                            response_list[0])]]

        for i, ele in enumerate(traj[2:]):
            if i%2==0:
                assert ele['role']=='user'
                obs_list = ele['content'].split('[SEP]')[2:]
                obs_list = ["'"+obs.strip()+"'" for obs in obs_list]
                observation_list.append(' [SEP] '.join(obs_list))
            else:
                assert ele['role']=='assistant'
                try:
                    action_list.append(split_thought_and_action(ele['content'])[1])
                    response_list.append(split_thought_and_action(ele['content'])[0])
                except:
                    pass_num+=1
                    break
                assert i//2 == len(response_list)-2
                assert len(observation_list)==len(action_list)
                final_data_list.append([uid, form_conversation(WEBSHOP_TEMPLATE.format(
                    task_description=task_description,step_count=i//2+1,history_length=min(i//2+1, history_length),action_history=form_history(i//2+1-min(i//2, history_length-1),observation_list[-1-history_length:-1],action_list[-1-history_length:-1]),
                    current_step=i//2+2,current_observation=observation_list[-1],available_actions=form_action_string(admissible_action_from_observation(observation_list[-1],action_list[-1]))),
                                            response_list[-1])]) 

        return final_data_list 

    # Load the dataset (all splits)
    dataset = load_dataset("CharlieDreemur/OpenManus-RL")

    def filter_trial_ids(example):
        return "webshop" in example.get("id", "").lower()

    # Apply filtering to every split
    train_trials = dataset["train"].filter(filter_trial_ids)

    output_dir = Path("./data/OpenManus-RL/")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = "./data/OpenManus-RL/webshop_train_sft.parquet"

    all_dataset = []
    for idx,data in enumerate(train_trials.select(range(2048+1024))):
        all_dataset.extend(split_one_traj_to_multiple_samples(idx,data['conversations']))

    df = pd.DataFrame(all_dataset, columns=["id", "messages"])
    df.to_parquet("./data/OpenManus-RL/webshop_train_sft.parquet", engine="pyarrow", index=False)
    print(f"{len(all_dataset)} examples, pass {pass_num}")

    all_dataset = []
    for idx,data in enumerate(train_trials.select(range(2048+1024,len(train_trials)))):
        all_dataset.extend(split_one_traj_to_multiple_samples(2048+1024+idx,data['conversations']))

    df = pd.DataFrame(all_dataset, columns=["id", "messages"])
    df.to_parquet("./data/OpenManus-RL/webshop_test_sft.parquet", engine="pyarrow", index=False)

    # Quick sanity check
    print(f"{len(all_dataset)} examples,{pass_num}")
    # Example: inspect one item
    print(all_dataset[0])
    print(all_dataset[-2])
    print(all_dataset[-1])

# process_webshop()


def process_search():
    dataset = load_dataset('ThornZ/Search-R1-SFT')




def print_data_length():
    import json
    from transformers import AutoTokenizer
    # dataset = pd.read_parquet('./data/OpenManus-RL/webshop_train_sft.parquet')
    with open('./data/vllm_sample/webshop_samples.jsonl') as f:
        dataset = [json.loads(line) for line in f]
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    i=0
    len_list = []
    # print(dataset['messages'].iloc[-1])
    # for data in dataset['messages']:
    for data in dataset:
        # if 'Think:' in data[1]['content']:
        #     print(data[1])
        # input_ids = tokenizer.apply_chat_template(data)
        input_ids = tokenizer.tokenize(data['text'])
        len_list.append(len(input_ids))
        if len(input_ids)>512:
            i+=1

    print(i,sum(len_list)/len(len_list),max(len_list))

# print_data_length()



def split_data_to_parts():
    import pandas as pd
    import numpy as np
    from pathlib import Path

    dataset = load_dataset("CharlieDreemur/OpenManus-RL")

    def filter_trial_ids(example):
        return "_trial_" in example.get("id", "").lower()

    # Apply filtering to every split
    trail_trails = dataset["train"].filter(filter_trial_ids)

    # Path to the parquet file
    path = Path("./data/OpenManus-RL/alfworld_4")

    trail_trails = trail_trails.select(range(len(trail_trails)//4))
    
    trail_trails = trail_trails.shuffle(seed=0)
    train_dataset = trail_trails.select(range(len(trail_trails)*4//5))
    test_dataset = trail_trails.select(range(len(trail_trails)*4//5,len(trail_trails)))

    print(f'train_dataset:Len{len(train_dataset)},test_dataset:Len{len(test_dataset)}')
    train_dataset.to_parquet(path/"train_sft.parquet")
    test_dataset.to_parquet(path/"test_sft.parquet")

    # Load parquet
    # df = pd.read_parquet(path)

    # # Split into 4 roughly equal parts
    # splits = np.array_split(df, 4)

    # # Save each split
    # for i, split_df in enumerate(splits, start=1):
    #     out_path = path.parent / f"train_sft_part{i}.parquet"
    #     split_df.to_parquet(out_path, index=False)

    # print("Done. Files saved in:", path.parent)



def show_mu_and_token():
    import json
    from transformers import AutoTokenizer
    def color_tokens_html(tokens, weights, base_rgb=(255, 0, 0),
                      min_alpha=0.05, max_alpha=0.85):

        assert len(tokens) == len(weights)

        w_min, w_max = min(weights), max(weights)
        denom = (w_max - w_min) if (w_max - w_min) != 0 else 1.0

        def norm(w):
            return (w - w_min) / denom

        r, g, b = base_rgb
        spans = []

        for tok, w in zip(tokens, weights):
            alpha = min_alpha + norm(w) * (max_alpha - min_alpha)
            span = (
                f'<span style="background: rgba({r},{g},{b},{alpha}); '
                f'padding:2px 4px; border-radius:4px;">{tok}</span>'
            )
            spans.append(span)

        return " ".join(spans)


    def save_colored_tokens(tokens, weights, filename="colored_tokens.html"):
        html_content = color_tokens_html(tokens, weights)

        full_html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>Colored Tokens</title>
        </head>
        <body style="font-family: Arial; line-height: 1.8;">
            {html_content}
        </body>
        </html>
        """

        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_html)

        print(f"Saved to {filename}")

    # Example
    data = pd.read_parquet("./data/OpenManus-RL/webshop_test_sft.parquet")
    mu = json.load(open("./data/OpenManus-RL/webshop_uncertainty_test.pt"))
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')

    tokens = tokenizer.apply_chat_template(data['messages'][0],tokenize=False,add_generation_prompt=False)
    tokens = tokenizer.tokenize(tokens)[1:]
    weights = mu[0]
    assert len(tokens)==len(weights),(len(tokens),len(weights))

    save_colored_tokens(tokens, weights)



def show_entropy_and_token():
    import json
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    # # def color_tokens_html(tokens, weights, base_rgb=(255, 0, 0),min_alpha=0.05, max_alpha=0.85):

    #     assert len(tokens) == len(weights)

    #     w_min, w_max = min(weights), max(weights)
    #     denom = (w_max - w_min) if (w_max - w_min) != 0 else 1.0

    #     def norm(w):
    #         return (w - w_min) / denom

    #     r, g, b = base_rgb
    #     spans = []

    #     for tok, w in zip(tokens, weights):
    #         alpha = min_alpha + norm(w) * (max_alpha - min_alpha)
    #         span = (
    #             f'<span style="background: rgba({r},{g},{b},{alpha}); '
    #             f'padding:2px 4px; border-radius:4px;">{tok}</span>'
    #         )
    #         spans.append(span)

    #     return " ".join(spans)
    def color_tokens_html(tokens, weights,
                        low_rgb=(0, 102, 255),      # blue
                        high_rgb=(255, 0, 0)):      # red

        assert len(tokens) == len(weights)

        w_min, w_max = min(weights), max(weights)
        denom = (w_max - w_min) if (w_max - w_min) != 0 else 1.0

        def norm(w):
            return (w - w_min) / denom

        spans = []

        for tok, w in zip(tokens, weights):
            t = norm(w)

            r = int(low_rgb[0] + t * (high_rgb[0] - low_rgb[0]))
            g = int(low_rgb[1] + t * (high_rgb[1] - low_rgb[1]))
            b = int(low_rgb[2] + t * (high_rgb[2] - low_rgb[2]))

            span = (
                f'<span style="background: rgb({r},{g},{b}); '
                f'padding:2px 4px; border-radius:4px; color:white;">{tok}</span>'
            )
            spans.append(span)

        return " ".join(spans)

    def save_colored_tokens(tokens, weights, filename="colored_tokens.html"):
        html_content = color_tokens_html(tokens, weights)

        full_html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>Colored Tokens</title>
        </head>
        <body style="font-family: Arial; line-height: 1.8;">
            {html_content}
        </body>
        </html>
        """

        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_html)

        print(f"Saved to {filename}")

    # Example
    data = pd.read_parquet("./data/OpenManus-RL/webshop_test_sft.parquet")
    with open("./data/vllm_sample/webshop_test_samples.jsonl") as f:
        gen = [json.loads(line) for line in f]
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    model1 = AutoModelForCausalLM.from_pretrained('/nobackup2/windy/verl-agent-master/checkpoints/256/qwen2.5_1.5b_webshop_sft/global_step_30')
    messages = data['messages'][0]
    messages = [{'role':'user','content':messages[0]['content'] + '\nThis is an example for a response to the question:\n'+ messages[1]['content'] + '\nNow answer with a response of your own, including the thinking process:'
                                                },messages[1]]
    tokens = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    inputs = tokenizer.apply_chat_template(
        [messages[0]],return_tensors="pt",add_generation_prompt=True
    )
    prompt_len = inputs.shape[1]
    inputs = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,return_tensors="pt",return_dict=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  
        probs = torch.nn.functional.softmax(logits, dim=-1)
        token_entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

        outputs = model1(**inputs)
        logits = outputs.logits  
        probs = torch.nn.functional.softmax(logits, dim=-1)
        token_entropy1 = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    tokens = tokenizer.tokenize(tokens)[prompt_len:]
    weights = token_entropy1[0,prompt_len-1:-1] - token_entropy[0,prompt_len-1:-1] 
    weights = (weights - weights.mean())/weights.std()
    assert len(tokens)==len(weights),(logits.shape,len(tokens),len(weights))

    save_colored_tokens(tokens, weights)

# show_entropy_and_token()

# split_data_to_parts()
import pandas as pd
import os

# paths
# input_path = "./data/OpenManus-RL/webshop_train_sft.parquet"
output_path = "./data/OpenManus-RL/webshop32/webshop_train_sft.parquet"
output_path_test = "./data/OpenManus-RL/webshop32/webshop_test_sft.parquet"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
# read parquet

df1 = pd.read_parquet('./data/OpenManus-RL/webshop_train_sft.parquet')
df2 = pd.read_parquet('./data/OpenManus-RL/webshop_test_sft.parquet')

# Combine the dataframes vertically (appending rows)
merged_df = pd.concat([df1, df2], ignore_index=True)
print(merged_df)

# df1 = pd.read_parquet('./data/OpenManus-RL/webshop16/train_sft.parquet')
# df2 = pd.read_parquet('./data/OpenManus-RL/webshop16/test_sft.parquet')

# # Combine the dataframes vertically (appending rows)
# merged_df = pd.concat([df1, df2], ignore_index=True)
# print(merged_df)
# raise ValueError

# take first 1/8 of rows
# n=5
n = len(merged_df) // (16 * 2)
# df_subset = merged_df.iloc[:n]
# df_subset = df_subset.sample(frac=1,random_state=0)
df_subset = merged_df.sample(frac=1,random_state=0)

# write to another folder
train_set = df_subset.iloc[:n*4//5]
# train_set = pd.concat([train_set for _ in range(8)] , ignore_index=True)
test_set = df_subset.iloc[n*4//5:n]
# test_set = pd.concat([test_set for _ in range(8)] , ignore_index=True)
train_set.to_parquet(output_path, index=False)
test_set.to_parquet(output_path_test, index=False)
df = pd.read_parquet(output_path)
# print(df)
print(len(df),len(set([df.iloc[i]['messages'][1]['content'] for i in range(len(df))])))
df = pd.read_parquet(output_path_test)
print(len(df),len(set([df.iloc[i]['messages'][1]['content'] for i in range(len(df))])))
