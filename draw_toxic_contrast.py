import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Data
approaches = ["GR\nPO", "GRPO\n+SEAL", "GIG\nPO", "GIGPO\n+SEAL"]
# toxicity_scores = [6.34, 4.22, 7.22, 4.31]
# toxicity_scores = [6.32, 3.22, 5.22, 4.10]
# toxicity_scores = [18.05, 12.23, 33.60, 26.50] # qwen7-gigpo on webshop
toxicity_scores = [21.05, 17.23, 25.60, 18.50] # llama3-gigpo on webshop

# Academic-style colors
base_color = "#800074"   # GRPO / GIGPO
seal_color = "#67d0d0"   # +SEAL variants

colors = [
    base_color if "SE" not in approach else seal_color
    for approach in approaches
]

ax = plt.gca()
ax.set_facecolor("#FAFAFA")

# Figure
plt.figure(figsize=(6, 4))

bars = plt.bar(
    approaches,
    toxicity_scores,
    color=colors,
    edgecolor="black",
    linewidth=0.8
)

# Labels and title
# plt.xlabel("Approach", fontsize=12)
plt.ylabel("Sent. Dup.", fontsize=28)
# plt.ylabel("Judge Score", fontsize=28)
plt.title("Llama3.2-3B/WebShop", fontsize=30)
# plt.title("Qwen2.5-7B/WebShop", fontsize=30)

# Axis formatting
plt.ylim(15, 26)
plt.xticks(fontsize=28)
plt.yticks(range(15,30,5),fontsize=28)
# plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)

# Add values above bars
hatches = [
        "/" if "SE" not in approach else "\\"
        for approach in approaches
]

for bar, score, hatch in zip(bars, toxicity_scores, hatches):
    # plt.text(
    #     bar.get_x() + bar.get_width() / 2,
    #     bar.get_height() + 0.02,
    #     f"{score:.2f}",
    #     ha="center",
    #     va="bottom",
    #     fontsize=10
    # )
    bar.set_hatch(hatch)



# Legend
base_patch = mpatches.Patch(
    facecolor=base_color,
    edgecolor="black",
    label="Base methods"
)
seal_patch = mpatches.Patch(
    facecolor=seal_color,
    edgecolor="black",
    label="+SEAL methods"
)

# plt.legend(
#     handles=[base_patch, seal_patch],
#     frameon=False,
#     fontsize=10,
#     loc="upper left"
# )

# Remove unnecessary borders for a cleaner academic style
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig('figs/toxic_contrast_llama.pdf',dpi=300)