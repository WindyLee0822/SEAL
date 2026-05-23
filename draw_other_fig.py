import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# -----------------------------
# Data -> target distribution p
# -----------------------------
real_distribution = torch.tensor([3,3,5,3,2,2,3,3,3,5,
                                  0.2,2,0.2,0.2,0.2,5,0,0,0,0,
                                  0,1,0,0,0,0], dtype = torch.float32)/2
real_distribution = real_distribution / real_distribution.sum()
real_dist = torch.distributions.Categorical(probs=real_distribution)
sample_time = 24
x = torch.zeros_like(real_distribution)
for _ in range(sample_time):
    a = real_dist.sample()
    x[a]+=1

p = x / x.sum()

# -----------------------------
# Alpha-divergence (Amari) as a torch loss
# -----------------------------
def alpha_divergence_torch(p, q, alpha: float, eps: float = 1e-12):
    """
    D_alpha(p||q):
      alpha != ±1:
        D = (4/(1-alpha^2)) * (1 - sum_i p_i^((1-alpha)/2) * q_i^((1+alpha)/2))
      alpha -> 1  : KL(p||q)
      alpha -> -1 : KL(q||p)
    """
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)


    if abs(alpha - 1.0) < 1e-8:
        return torch.sum((torch.log(q) - torch.log(p)))
    if abs(alpha) < 1e-8:
        return torch.sum(q/p * (torch.log(q) - torch.log(p)))
    x = p/q
    return 1/x * (x**alpha - alpha *x - (1-alpha))/(alpha*(alpha-1))

# -----------------------------
# Parameterization: logits -> q via softmax
# -----------------------------
alpha = 0.99  # try 0.0, 0.5, -0.5, 1.0 (KL), -1.0 (reverse KL)
steps = 2000

logits = torch.zeros_like(p, requires_grad=True)
opt = torch.optim.Adam([logits], lr=0.2)

loss_hist = []

def collect_batch_and_update(sampling_policy,optimizer,logits):
    # Storage
    logps = []
    old_logps = []


    # Rollout (bandit-like: same "state" each step)BATCH_SIZE
    BATCH_SIZE=20
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
        # entropies.append(-(probs * (probs + 1e-9).log()).sum())  # H(pi)
        # Z_p.append((probs/sample_probs).sum())


    # Tensors
    logps = torch.stack(logps)             # [T]
    old_logps = torch.stack(old_logps)
    # Advantage normalization (often stabilizes training)
    # if True:
    #     adv_mean, adv_std = advantages.mean(), advantages.std().clamp_min(1e-8)
    #     advantages = (advantages - adv_mean) #/ adv_std

    policy_loss = alpha_divergence_torch(old_logps.exp(),logps.exp(),alpha)
    # Z_a = final_rewards.exp().sum()

    # policy_loss = -(advantages.detach() * logps) #GRPO
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
    # returns = (torch.softmax(logits,dim=-1) * reward_probs ).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

RLTURN=2000
# RL on Pre-trained Model
pretrain_conv =[]
for _ in range(RLTURN):
    pretrain_conv.append(collect_batch_and_update(p.detach().clone(),opt,logits))


# print(f"Final loss D_alpha(p||q), alpha={alpha}: {loss_hist[-1]:.8f}")

with torch.no_grad():
    q_opt = F.softmax(logits, dim=0).cpu()
    p_cpu = p.cpu()

print("p (target):", torch.round(p_cpu * 10000) / 10000)
print("q (opt)   :", torch.round(q_opt * 10000) / 10000)

# -----------------------------
# Plot: target vs optimized
# -----------------------------
idx = torch.arange(len(p_cpu)).numpy()

plt.figure()
plt.plot(idx, real_distribution.numpy(), marker="*", label="real p")
# plt.plot(idx, p_cpu.numpy(), marker="o", label="sampled p")
plt.bar(idx,p_cpu.numpy(),color='orange',label="sampled p")
plt.plot(idx, q_opt.numpy(), marker="x", color='violet',label="optimized q")
plt.title(f"Target vs optimized distribution (alpha={alpha})")
plt.xlabel("Index")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.savefig('figs/div.pdf')

# Optional: plot loss curve
# plt.figure()
# plt.plot(loss_hist)
# plt.title("Optimization loss")
# plt.xlabel("Step")
# plt.ylabel("D_alpha(p||q)")
# plt.tight_layout()
# plt.show()
