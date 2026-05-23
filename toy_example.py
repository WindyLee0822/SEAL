import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from cycler import cycler

torch.manual_seed(2)
torch.cuda.manual_seed_all(2)

# Define academic purple-orange palette
academic_palette = ['#4B2E83', '#D97706']

# Apply globally
plt.rcParams['axes.prop_cycle'] = cycler(color=academic_palette)

# Optional journal-style tweaks
# plt.rcParams.update({
#     'font.size': 11,
#     'axes.edgecolor': '#111827',
#     'axes.labelcolor': '#111827',
#     'xtick.color': '#111827',
#     'ytick.color': '#111827',
#     'text.color': '#111827',
#     'figure.facecolor': 'white',
#     'axes.facecolor': '#F3F4F6',
# })

torch.manual_seed(0)


def gaussian_peak(n: int, center: float, sigma: float, amp: float = 1.0, device=None):
    """
    1D Gaussian bump of length n centered at `center` (float ok), width `sigma`, amplitude `amp`.
    """
    x = torch.arange(n, device=device, dtype=torch.float32)
    return amp * torch.exp(-0.5 * ((x - center) / sigma) ** 2)


# initialize logits
n = 50
sigma1 = 3.0  # width of each peak (tune as you like)
v1 = gaussian_peak(n, 15, sigma1) + gaussian_peak(n, 35, sigma1)
v1 /= v1.max() 

logits = nn.Parameter(v1.detach().clone())  # shape [100]
optimizer = optim.Adam([logits], lr=2e-3)
sft_logits = nn.Parameter(v1.detach().clone())  # shape [100]
sft_optimizer = optim.Adam([sft_logits], lr=2e-3)
our_logits = nn.Parameter(v1.detach().clone())  # shape [100]
our_optimizer = optim.Adam([our_logits], lr=2e-3)

old_probs =  torch.softmax(v1,dim=-1)
old_logprobs = torch.log(old_probs)

# rewards 
num_peaks = 3
centers = [10,25,40]
sigmas = [5,5,5]
envelope = [4,5,4] 
v2 = torch.zeros(n, dtype=torch.float32)
for c, a, sigma in zip(centers, envelope, sigmas):
    v2 += gaussian_peak(n, float(c), sigma, amp=float(a))
reward_probs = (v2 - v2.min()) /(v2.max()-v2.min())
reward_probs = torch.softmax(reward_probs,dim=-1)

# Sample SFT data
percentile = 0.5
num_samples = 50  # choose how many indices you want to sample
threshold = torch.quantile(v2, percentile)
valid_idx = torch.nonzero(v2 > threshold, as_tuple=True)[0]
sampled_indices = torch.multinomial(
    torch.softmax(reward_probs[valid_idx],dim=-1),
    num_samples=num_samples,
    replacement=True
)
sampled_indices = valid_idx[sampled_indices]
valid_samples = sampled_indices[:len(sampled_indices)//5]
sampled_indices = sampled_indices[len(sampled_indices)//5:]
num_samples = num_samples//5*4
group_number = torch.zeros(n)
for i in sampled_indices:
    group_number[i]+=1



BATCH_SIZE=16
HYPER = 1
def kl_p2_p1_from_logits(z1, p2, eps=1e-12):
    p2 = normalize_probs(p2, eps=eps)
    log_p1 = F.log_softmax(z1, dim=-1)            # stable log p1
    log_p2 = torch.log(p2)                         # stable since clamped
    kl = (p2 * (log_p2 - log_p1)).sum(dim=-1)      # [B]
    return kl.mean()

def f(x,alpha=0):
    return (x**alpha - alpha*x - (1-alpha))/(alpha*(alpha-1))

def SFT(sample_indices,logits,optimizer,ours=False,epochs=2000,alpha=0.):
    best_valid_loss = 1e9
    fail_time = 0
    for _ in range(epochs):
        for i in range(len(sampled_indices)//BATCH_SIZE+1):
            batch_data = sampled_indices[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            
            probs = torch.softmax(logits, dim=0)
            logprobs = torch.log(probs)
            if len(batch_data)==0:break
            
            if ours:
                probs = probs.repeat(len(batch_data), 1)
                # q = old_probs.repeat(len(batch_data), 1)
                q = torch.randn((probs.shape[0],probs.shape[1])) * 1e-6
                q -= (q.min() - 1e-6)
                q[torch.arange(q.shape[0]), batch_data] += 1
                q = q/q.sum(-1).unsqueeze(-1)
                # values, indices = torch.topk(q, 5, dim=-1)
                # q = values/values.sum(-1).unsqueeze(-1)
                # probs = probs.gather(1,indices);probs = probs/probs.sum(-1).unsqueeze(-1)
                if alpha==0:
                    loss = probs/q * (- torch.log(q/probs))
                else:
                    x = q/probs
                    loss = ((x**(alpha) - alpha*x  - (1-alpha))/(alpha*(alpha-1))) * probs
            else:
                loss = -logprobs[batch_data]
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for i in range(len(valid_samples)//BATCH_SIZE+1):
                batch_data = valid_samples[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                probs = torch.softmax(logits, dim=0)
                logprobs = torch.log(probs)
                if len(batch_data)==0:break
                
                if ours:
                    probs = probs.repeat(len(batch_data), 1)
                    # q = old_probs.repeat(len(batch_data), 1)
                    q = torch.randn((probs.shape[0],probs.shape[1])) * 1e-6
                    q -= (q.min() - 1e-6)
                    q[torch.arange(q.shape[0]), batch_data] += 1
                    q = q/q.sum(-1).unsqueeze(-1)
                    # values, indices = torch.topk(q, 5, dim=-1)
                    # q = values/values.sum(-1).unsqueeze(-1)
                    # probs = probs.gather(1,indices);probs = probs/probs.sum(-1).unsqueeze(-1)
                    if alpha==0:
                        loss = probs/q * (- torch.log(q/probs))
                    else:
                        x = q/probs
                        loss = ((x**(alpha) - alpha*x  - (1-alpha))/(alpha*(alpha-1))) * probs
                else:
                    loss = -logprobs[batch_data].mean()
            loss = loss.mean()
            if best_valid_loss < loss:
                fail_time +=1
                if fail_time >=3:
                    break
            else:
                fail_time = 0
        

            
        
        


# SFT
SFT(sampled_indices,sft_logits,sft_optimizer)
sft_copy = sft_logits.detach().clone()
# SFT of ours
SFT(sampled_indices,our_logits,our_optimizer,True)
our_copy = our_logits.detach().clone()



BATCH_SIZE=32
def collect_batch_and_update(sampling_policy,optimizer,logits):
    # Storage
    logps = []
    rewards = []
    values = []
    entropies = []
    old_logps = []
    Z_p = []

    # Rollout (bandit-like: same "state" each step)
    for _ in range(BATCH_SIZE):
        probs = torch.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs=probs)
        sample_probs_temp = torch.softmax(sampling_policy/2, dim=0)
        sample_probs = torch.softmax(sampling_policy, dim=0)
        # sample_probs = sampling_policy
        sample_dist_temp = torch.distributions.Categorical(probs=sample_probs_temp)
        sample_dist = torch.distributions.Categorical(probs=sample_probs)
        a = sample_dist_temp.sample()
        logps.append(dist.log_prob(a))
        # old_logps.append(old_logprobs[int(a)])
        old_logps.append(sample_dist.log_prob(a))
        entropies.append(-(probs * (probs + 1e-9).log()).sum())  # H(pi)
        Z_p.append((probs/sample_probs).sum())

        r = reward_probs[int(a)]           # scalar reward from the reward curve
        rewards.append(r.detach())


    # Tensors
    logps = torch.stack(logps)             # [T]
    old_logps = torch.stack(old_logps)
    rewards = torch.stack(rewards)         # [T]
    Z_p = torch.stack(Z_p)

    # Bootstrap with V(s_{T}) = value_param (state is same)
    advantages = rewards
    # Advantage normalization (often stabilizes training)
    if True:
        adv_mean, adv_std = advantages.mean(), advantages.std().clamp_min(1e-8)
        advantages = (advantages - adv_mean) #/ adv_std


    # Z_a = final_rewards.exp().sum()

    policy_loss = -(advantages.detach() * logps) #GRPO
    # policy_loss = logps.exp()/old_logps.exp() * (logps - old_logps.detach() - advantages.detach()) 
    # policy_loss = -advantages.exp() * logps # rKL doesn't work well
    # policy_loss = advantages.exp() * torch.abs(logps.exp()/old_logps.exp()/advantages.exp() - 1) #tv
    # x = (logps - old_logps.detach() - advantages.detach()).exp()
    # policy_loss = advantages.exp() * (x*torch.log(x) - (x+1)*torch.log((x+1)/2))/2 #JS
    # policy_loss = (logps.exp()/old_logps.exp()).detach() * (logps - old_logps - advantages + reward_probs.exp().sum().log())**2. #FLOWRL
    # x = ((logps - old_logps.detach() - advantages.detach())/2).exp()
    # policy_loss = advantages.exp() * ((x-1)**2)
    # policy_loss = advantages.exp() * (x-1)*x.log() 
    # policy_loss = -(advantages.detach() * logps)*(logps.exp()/old_logps.exp()).detach() + (logps - old_logps) #GRPO with KL

    # policy_loss = advantages.exp()/Z_a * torch.abs(logps.exp()/old_logps.exp()/advantages.exp() * Z_p/Z_a - 1) # tv_with_Z
    loss = policy_loss.mean()
    returns = (torch.softmax(logits,dim=-1) * reward_probs ).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return returns.clone().item()

RLTURN=2000
# RL on Pre-trained Model
pretrain_conv =[]
for _ in tqdm(range(RLTURN)):
    pretrain_conv.append(collect_batch_and_update(logits.detach().clone(),optimizer,logits))


# RL on SFT model
sft_conv =[]
for _ in tqdm(range(RLTURN)):
    sft_conv.append(collect_batch_and_update(sft_logits.detach().clone(),sft_optimizer,sft_logits))


# RL on Our Model
our_conv =[]
for _ in tqdm(range(RLTURN)):
    our_conv.append(collect_batch_and_update(our_logits.detach().clone(),our_optimizer,our_logits))


# with torch.no_grad():
#     final_policy = torch.softmax(logits, dim=0).cpu()
#     sampling_policy = torch.softmax(sampling_policy, dim=0).cpu()

# var = (sampling_policy * (reward_probs - reward_probs.mean()) ** 2).sum()
# std = torch.sqrt(var)
# final_rewards = (reward_probs - (reward_probs*sampling_policy).sum())/(std)
plt.figure(figsize=(6, 4))
plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
})
plt.plot(range(50), (old_probs).numpy(),color='black',label="ref.",lw=2)
plt.plot(range(50), (sft_copy.exp()/(sft_copy.exp()).sum()).detach().numpy(), color='blue',label=f"sft",lw=2)
plt.plot(range(50), (our_copy.exp()/(our_copy.exp()).sum()).detach().numpy(), color='violet',label=f"ours",lw=2)
plt.plot(range(50), (reward_probs).numpy(),color='orange',label="reward",lw=2)
plt.legend()
plt.savefig('figs/distribution.pdf')

plt.figure(figsize=(6, 4))
plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
})
plt.plot(range(50), (sft_logits.exp()/(sft_logits.exp()).sum()).detach().numpy(), color='red',label=f"sft+rl",lw=1)
plt.plot(range(50), (our_logits.exp()/(our_logits.exp()).sum()).detach().numpy(), color='pink',label=f"ours+rl",lw=1)
plt.plot(range(50), (reward_probs).numpy(),color='orange',label="reward",lw=2)
print(sft_logits.argmax(),our_logits.argmax(),reward_probs.argmax())
plt.legend()
plt.savefig('figs/rl_distribution.pdf')

plt.figure(figsize=(6, 4))
plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
})
plt.plot(range(RLTURN), (torch.tensor(pretrain_conv)).numpy(),color='black',label="rl",lw=2)
plt.plot(range(RLTURN), (torch.tensor(sft_conv)).numpy(),color='blue',label="sft+rl",lw=2)
plt.plot(range(RLTURN), (torch.tensor(our_conv)).numpy(),color='violet',label="our+rl",lw=2)
# plt.ylim(0.32,0.36)
# for i in range(len(sft_conv)):
#     if sft_conv[i]==sft_conv[-1]:print('sft',i)
#     if our_conv[i]==our_conv[-1]:print('our',i)
plt.legend()
plt.savefig('figs/convergence.pdf')




