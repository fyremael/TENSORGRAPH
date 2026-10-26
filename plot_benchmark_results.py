"""
Script to plot full-coverage benchmark graphs for TENSORGRAPH Colab T4 GPU results.
Saves dark-mode PNG charts to artifact directory.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Set dark theme style
plt.style.use('dark_background')
fig_color = '#0F172A'
ax_color = '#1E293B'
text_color = '#F8FAFC'

# Artifact output directory
artifact_dir = r"C:\Users\jamie\.gemini\antigravity\brain\6cb6e08e-7d6d-448b-9e30-8e1012ab2b55"

# -----------------------------------------------------------------------------
# CHART 8: Extended Application Domains (DiT, MoE, GNN)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.5), facecolor=fig_color)
ax.set_facecolor(ax_color)

categories = [
    'Diffusion Transformers (DiT)\n(AdaLN + SwiGLU Block)',
    'Mixture-of-Experts (MoE)\n(8-Expert Gated Routing)',
    'Graph Neural Networks (GNN)\n(Scatter-Sum Message Passing)'
]

eager_us = [654.16, 893.33, 550.64]
tensorgraph_us = [567.31, 792.91, 465.69]

x = np.arange(len(categories))
width = 0.35

rects1 = ax.bar(x - width/2, eager_us, width, label='PyTorch Eager', color='#EF4444', edgecolor='white', alpha=0.9)
rects2 = ax.bar(x + width/2, tensorgraph_us, width, label='TENSORGRAPH Optimized', color='#10B981', edgecolor='white', alpha=0.9)

ax.set_title('TENSORGRAPH Extended Application Domains Performance (Tesla T4)', fontsize=13, fontweight='bold', pad=15, color=text_color)
ax.set_ylabel('Latency (µs) [Lower is Better]', fontsize=12, fontweight='bold', color=text_color)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax.legend(frameon=True, facecolor='#334155', edgecolor='none', labelcolor=text_color, fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.3)

speedups = [eager_us[i] / tensorgraph_us[i] for i in range(len(categories))]

for i in range(len(categories)):
    ax.annotate(f'{tensorgraph_us[i]:.1f} µs\n({speedups[i]:.2f}x Speedup)',
                xy=(x[i] + width/2, tensorgraph_us[i]),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#10B981')

plt.tight_layout()
chart8_path = os.path.join(artifact_dir, "extended_domains_chart.png")
plt.savefig(chart8_path, dpi=200, facecolor=fig_color)
plt.close()

print("Extended domains chart successfully generated and saved to artifact directory!")
