import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 14})
# # Data
# models = [
#     "LLaVA-One-Vision",
#     "LLaVA-Interleave",
#     "DeepSeek",
#     "Mantis",
#     "ChatGPT-4o",
#     "ResNet + VGGFace",
#     "ResNet + WebFace"
# ]

# label_0 = [0.9916, 1, 0.9356, 1, 0.8863, 0.949, 0.898]
# label_1 = [0.6913, 0, 0.0227, 0.0016, 0.9997, 0.999, 0.9983]

# # Compute the bias towards label 1
# bias_towards_label1 = [l1 - l0 for l1, l0 in zip(label_1, label_0)]

# # Plot horizontal bar chart
# plt.figure(figsize=(12, 6))
# bars = plt.barh(models, bias_towards_label1, color=['green' if x > 0 else 'red' for x in bias_towards_label1])
# plt.axvline(0, color='gray', linestyle='--')

# # Annotate bars
# for bar in bars:
#     plt.text(
#         bar.get_width() + (0.01 if bar.get_width() >= 0 else -0.05),
#         bar.get_y() + bar.get_height() / 2,
#         f"{bar.get_width():.2f}",
#         va='center',
#         color='black'
#     )

# # Plot settings
# # plt.title("Bias Toward Label 1 (Difference) Compared to Label 0 (Similarity)")
# plt.xlabel("acc(Label 1) - acc(Label 0)", labelpad=20)  # tăng khoảng cách với biểu đồ
# plt.ylabel("Model", labelpad=15)
# plt.tight_layout()

# # Save figure
# plt.savefig("model_bias_label1_vs_label0.pdf", dpi=300, bbox_inches='tight')  # You can also save as .pdf or .svg
####################################################################### Tạo nhóm dữ liệu theo từng mô hình để hỗ trợ stacked bar chart
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

models_with_both = [
    "Llava One vision",
    "Llava Interleave",
    "Deepseek",
    "Mantis"
]

acc0_exp = [0.892, 0.998, 0.9166, 0.9996]
acc1_exp = [0.8726, 0.185, 0.119, 0.057]
acc0_dir = [0.9916, 1, 0.9356, 1]
acc1_dir = [0.6913, 0, 0.0227, 0.0016]

models_single = ["ChatGPT 4o", "ResNet + VGGFace", "ResNet + WebFace"]
acc0_single = [0.8863, 0.949, 0.898]
acc1_single = [0.9997, 0.999, 0.9983]

bar_width = 0.25
gap_between_label = 0.05
gap_between_model = 0.8

fig, ax = plt.subplots(figsize=(14, 6))

x = []
labels = []

def color_with_alpha(color, alpha=1.0):
    import matplotlib.colors as mcolors
    c = mcolors.to_rgba(color, alpha)
    return c

for i, model in enumerate(models_with_both):
    base_pos = i * (2 * bar_width + gap_between_label + gap_between_model)

    # Blue for Label 0, orange for Label 1
    blue = "#1f77b4"
    orange = "#ff7f0e"

    # Label 0 bars (Direct lighter, Explanation darker)
    ax.bar(base_pos, acc0_dir[i], width=bar_width, color=color_with_alpha(blue, 0.4))
    ax.bar(base_pos, acc0_exp[i], width=bar_width, bottom=acc0_dir[i], color=color_with_alpha(blue, 1.0))

    # Label 1 bars (Direct lighter, Explanation darker)
    pos_label1 = base_pos + bar_width + gap_between_label
    ax.bar(pos_label1, acc1_dir[i], width=bar_width, color=color_with_alpha(orange, 0.4))
    ax.bar(pos_label1, acc1_exp[i], width=bar_width, bottom=acc1_dir[i], color=color_with_alpha(orange, 1.0))

    # X positions for ticks at midpoint between label0 and label1 bars
    x.append(base_pos + bar_width + gap_between_label / 2)
    labels.append(model)

start_single = len(models_with_both) * (2 * bar_width + gap_between_label + gap_between_model)
for i, model in enumerate(models_single):
    pos0 = start_single + i * (2 * bar_width + gap_between_model)
    pos1 = pos0 + bar_width + gap_between_label

    # Just two bars, no stacking, using color for label 0 and 1
    ax.bar(pos0, acc0_single[i], width=bar_width, color=color_with_alpha(blue, 1.0))
    ax.bar(pos1, acc1_single[i], width=bar_width, color=color_with_alpha(orange, 1.0))

    x.append(pos0 + bar_width / 2 + gap_between_label / 2)
    labels.append(model)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel("Accuracy")
# ax.set_title("Verification Accuracy by Model and Label")

# Legend for Label 0 and Label 1 only
legend_elements = [
    Patch(facecolor=blue, label="Label 0 (same)"),
    Patch(facecolor=orange, label="Label 1 (different)"),
]

ax.legend(handles=legend_elements, loc='upper left')

# Add mini legend box inside plot for explanation vs direct
from matplotlib.lines import Line2D

direct_patch = Patch(facecolor=color_with_alpha("gray", 0.4), label='Direct return (lighter)')
explanation_patch = Patch(facecolor=color_with_alpha("gray", 1.0), label='Explanation-based (darker)')

# Place mini legend
props = dict(boxstyle='round', facecolor='white', alpha=0.7)
ax.text(0.98, 0.05, 'Color Intensity:\nLighter = Direct return\nDarker = Explanation-based',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
# plt.show()
plt.savefig("direct_explaination.pdf", dpi=300, bbox_inches='tight')  # You can also save as .pdf or .svg

