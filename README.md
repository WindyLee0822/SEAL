
<h3 align="center">
<b>Entropy Dynamics in Agent Reinforcement Learning</b>
<br>
</h3>


<!-- <p align="center">
  <a href="https://arxiv.org/abs/2505.10978">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  &nbsp;
  <a href="https://github.com/langfengQ/verl-agent">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub Project"></a>
  &nbsp;
  <a href="https://huggingface.co/collections/langfeng01/verl-agent-684970e8f51babe2a6d98554">
    <img src="https://img.shields.io/badge/HuggingFace-Models-yellow?style=flat-square&logo=huggingface" alt="HuggingFace Models"></a>
  &nbsp;
  <a href="https://x.com/langfengq/status/1930848580505620677">
    <img src="https://img.shields.io/badge/Twitter-Channel-000000?style=flat-square&logo=x" alt="X Channel"></a>
</p> -->

## Introduction

Agentic large language models are increasingly used to solve real-world tasks by reasoning over goals, invoking tools, and interacting with external environments. Reinforcement learning provides a natural framework for improving these behaviors, and recent agent RL methods have achieved strong results across domains. However, the training dynamics of agent RL remain poorly understood, limiting our ability to diagnose instabilities and design more effective training algorithms.
In this work, we identify a previously underexplored phenomenon in agent RL, which we term \emph{cyclical entropy eruption}. Unlike single-turn reasoning RL, where entropy typically collapses and stays low, agent RL training exhibits unique recurring cycles of sharp entropy eruption and gradual subsidence. We decompose this dynamic into three phases and provide theoretical and empirical analyses of each, explaining the mechanisms underlying its cyclical oscillation. We further show that degenerate patterns such as sentence duplication and hallucination, once acquired during eruption, can persist and accumulate across cycles. Motivated by these findings, we propose SEAL (Separation-Enhanced Agent Learning), a lightweight auxiliary loss that separates correct and incorrect trajectories in representation space, directly targeting the root cause of entropy eruption. Experiments across multiple benchmarks, models, and RL algorithms demonstrate that SEAL stabilizes training and yields stronger downstream agent performance.



## Install Supported Environments

### 1. ALFWorld
Install with pip:
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

Use `--extra` to download pre-trained checkpoints and seq2seq data.

Play a Textworld game:
```bash
alfworld-play-tw
```
---

### 2. WebShop
WebShop requires Python <=3.10, so begin by creating a new `verl-agent-webshop` environment
```bash
conda create -n verl-agent-webshop python==3.10 -y
conda activate verl-agent-webshop
```

Install WebShop
```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

Note: If you encounter issues with gdown, you may need to visit `https://drive.google.com/`, get your Google Drive cookie, and paste it into `.cache/gdown/cookies.txt`.
Or you may need to manually download the files.

After WebShop is installed, return to the root directory of the repository and install the verl package in `verl-agent`:
```bash
cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
```
The warnings can be safely ignored.

---

### 3. Search
```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

Prepare dataset (data will be saved at `~/data/searchR1_processed_direct`):
```bash
cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py
```


Since faiss-gpu is not available via pip, we setup a separate conda environment for the local retrieval server. Running this server will use around 6GB of GPU memory per GPU, so make sure to account for this in your training run configuration. Build Retriever environments:
```bash
# Create and activate the retriever environment with Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch (with GPU support) and related libraries
conda install numpy==1.26.4 # needed to stop incompatible version of numpy from being installed via pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# Install the GPU version of faiss
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y

# Install the API service framework
pip install uvicorn fastapi
```

Download the index:
```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

Start the local flat e5 retrieval server: 
```bash
conda activate retriever

# redirect the output to a file to avoid cluttering the terminal
# we have observed outputting to the terminal causing spikes in server response times
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 
```


## SEAL

```bash
bash examples/gigpo_trainer/run_alfworld.sh # ALFWorld

bash examples/gigpo_trainer/run_webshop.sh # WebShop
```

The following are some arguments that you might need to adjust during experiments
```
algorithm.adv_estimator=gigpo # gigpo or grpo
actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct # model path
actor_rollout_ref.actor.optim.lr=2e-6  # learning rate
+actor_rollout_ref.actor.porj_layer=3  # the layer number of value head
+actor_rollout_ref.actor.classification_loss_coef=0.02 # weight of classification loss
```


# Acknowledgement

`verl-agent` codebase is built upon [GIGPO](https://github.com/langfengQ/verl-agent), [FlowRL](https://github.com/Xuekai-Zhu/FlowRL) and [veRL](https://github.com/volcengine/verl). The supported environments are adapted from [ALFWorld](https://github.com/alfworld/alfworld), [Search-R1](https://github.com/PeterGriffinJin/Search-R1),and [WebShop](https://github.com/princeton-nlp/WebShop). We sincerely thank the authors and contributors of these projects for their valuable work and for making their resources publicly available.


<!--- Citation
If you find `verl-agent` and `GiGPO` useful in your research or applications, we would appreciate it if you could cite our work:

```
@article{feng2025group,
  title={Group-in-Group Policy Optimization for LLM Agent Training},
  author={Feng, Lang and Xue, Zhenghai and Liu, Tingcong and An, Bo},
  journal={arXiv preprint arXiv:2505.10978},
  year={2025}
}
```

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=langfengQ/verl-agent&type=Date)](https://www.star-history.com/#langfengQ/verl-agent&Date) -->
