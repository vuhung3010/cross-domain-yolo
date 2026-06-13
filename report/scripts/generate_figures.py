#!/usr/bin/env python3
from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt

ROOT = Path('/home/kacchan/VuHung/project/nhandang_project_2')
RUNS = ROOT / 'runs' / 'train'
OUT = Path(__file__).resolve().parents[1] / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

MAIN = [
    ('A1', 'Source-only YOLO-G', 'sanity_pr1_baseline'),
    ('A2', 'YOLO-G image-level DA', 'city_foggycity_advgrl_faithful'),
    ('A3', 'AdvGRL', 'city_foggycity_advgrl_faithful_alpha080'),
    ('A4', 'RainMix-triplet', 'city_foggycity_advgrl_faithful_triplet_50ep'),
    ('A5', 'AdvGRL + RainMix-triplet', 'city_foggycity_advgrl_faithful_full_12'),
    ('A6', 'AdvGRL + multi-scale neck', 'city_foggycity_advgrl_faithful_neck_all_alpha075_50ep'),
    ('A7', 'Foreground-gated neck AdvGRL', 'city_foggycity_advgrl_faithful_neck_all_alpha075_objgate_50ep'),
]

SWEEP = [
    ('0.65', 'city_foggycity_advgrl_faithful_neck_all_alpha065_50ep'),
    ('0.70', 'city_foggycity_advgrl_faithful_neck_all_alpha070_50ep'),
    ('0.75', 'city_foggycity_advgrl_faithful_neck_all_alpha075_50ep'),
    ('0.80', 'city_foggycity_advgrl_faithful_neck_all_alpha080_50ep'),
    ('0.85', 'city_foggycity_advgrl_faithful_neck_all_alpha085_50ep'),
]

CURVES = [
    ('Source-only YOLO-G', 'sanity_pr1_baseline'),
    ('YOLO-G image-level DA', 'city_foggycity_advgrl_faithful'),
    ('AdvGRL', 'city_foggycity_advgrl_faithful_alpha080'),
    ('Neck AdvGRL', 'city_foggycity_advgrl_faithful_neck_all_alpha075_50ep'),
    ('Foreground-gated neck AdvGRL', 'city_foggycity_advgrl_faithful_neck_all_alpha075_objgate_50ep'),
]

def read_results(run):
    p = RUNS / run / 'results.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open(newline='') as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    return rows

def col(row, name):
    if name in row:
        return float(row[name])
    for k, v in row.items():
        if k.strip() == name:
            return float(v)
    raise KeyError(name)

def final_metrics(run):
    row = read_results(run)[-1]
    return {
        'precision': col(row, 'metrics/precision'),
        'recall': col(row, 'metrics/recall'),
        'map50': col(row, 'metrics/mAP_0.5'),
        'map5095': col(row, 'metrics/mAP_0.5:0.95'),
    }

def save(fig, name):
    fig.savefig(OUT / f'{name}.png')
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)

# Figure 1: main ablation bar chart
labels, map50, map5095 = [], [], []
for aid, label, run in MAIN:
    m = final_metrics(run)
    labels.append(f'{aid}\n{label}')
    map50.append(m['map50'])
    map5095.append(m['map5095'])

fig, ax = plt.subplots(figsize=(10, 4.8))
x = range(len(labels))
w = 0.38
ax.bar([i - w/2 for i in x], map50, width=w, label='mAP@0.5', color='#4C78A7')
ax.bar([i + w/2 for i in x], map5095, width=w, label='mAP@0.5:0.95', color='#F58518')
ax.axhline(map50[0], color='#4C78A7', linestyle='--', linewidth=1, alpha=0.7)
ax.set_ylabel('Score')
ax.set_title('Main ablation results on Foggy Cityscapes')
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=25, ha='right')
ax.set_ylim(0, max(map50) + 0.08)
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.25)
for i, v in enumerate(map50):
    ax.text(i - w/2, v + 0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
save(fig, 'fig_ablation_main')

# Figure 2: neck alpha sweep
alphas, sweep_map50, sweep_map5095 = [], [], []
for a, run in SWEEP:
    m = final_metrics(run)
    alphas.append(float(a))
    sweep_map50.append(m['map50'])
    sweep_map5095.append(m['map5095'])
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(alphas, sweep_map50, marker='o', linewidth=2, label='mAP@0.5', color='#4C78A7')
ax.plot(alphas, sweep_map5095, marker='s', linewidth=2, label='mAP@0.5:0.95', color='#F58518')
best_i = max(range(len(sweep_map50)), key=lambda i: sweep_map50[i])
ax.scatter([alphas[best_i]], [sweep_map50[best_i]], s=90, facecolors='none', edgecolors='#E45756', linewidth=2)
ax.annotate('best verified', (alphas[best_i], sweep_map50[best_i]), xytext=(8, 10), textcoords='offset points')
ax.set_xlabel('Adaptive-gradient threshold alpha')
ax.set_ylabel('Score')
ax.set_title('Multi-scale neck AdvGRL alpha sweep')
ax.set_ylim(0.24, 0.49)
ax.legend(frameon=False)
ax.grid(alpha=0.25)
save(fig, 'fig_neck_alpha_sweep')

# Figure 3: training curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for label, run in CURVES:
    rows = read_results(run)
    epochs = [int(float(r.get('epoch', i))) + 1 for i, r in enumerate(rows)]
    axes[0].plot(epochs, [col(r, 'metrics/mAP_0.5') for r in rows], label=label, linewidth=1.8)
    axes[1].plot(epochs, [col(r, 'metrics/mAP_0.5:0.95') for r in rows], label=label, linewidth=1.8)
axes[0].set_title('mAP@0.5 over training')
axes[1].set_title('mAP@0.5:0.95 over training')
for ax in axes:
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.grid(alpha=0.25)
axes[1].legend(frameon=False, loc='lower right')
save(fig, 'fig_training_curves')

# Figure 4: precision-recall scatter
fig, ax = plt.subplots(figsize=(6.8, 5))
for aid, label, run in MAIN:
    m = final_metrics(run)
    ax.scatter(m['recall'], m['precision'], s=80)
    ax.text(m['recall'] + 0.003, m['precision'] + 0.003, aid, fontsize=10)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-recall trade-off across ablations')
ax.grid(alpha=0.25)
ax.set_xlim(0.25, 0.44)
ax.set_ylim(0.64, 0.80)
save(fig, 'fig_precision_recall')

# Figure 5: DA diagnostics if available
run = 'city_foggycity_advgrl_faithful_neck_all_alpha075_50ep'
p = RUNS / run / 'da_losses.csv'
if p.exists():
    with p.open(newline='') as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    if rows:
        def get_series(names):
            for name in names:
                if any(k.strip() == name for k in rows[0].keys()):
                    return [float(next(v for k, v in r.items() if k.strip() == name)) for r in rows]
            return None
        step = list(range(1, len(rows) + 1))
        lam = get_series(['lambda_adv'])
        lc = get_series(['L_c', 'l_c'])
        da = get_series(['loss_da_image'])
        scale = get_series(['da_scale'])
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        series = [(lam, 'Adaptive weight', '#4C78A7'), (lc, 'Classifier probe loss', '#F58518'), (da, 'Image-level adaptation loss', '#54A24B'), (scale, 'DA scale', '#B279A2')]
        for ax, (y, title, color) in zip(axes.ravel(), series):
            if y is None:
                ax.text(0.5, 0.5, 'not recorded', ha='center', va='center', transform=ax.transAxes)
            else:
                stride = max(1, len(y) // 1200)
                ax.plot(step[::stride], y[::stride], color=color, linewidth=1)
            ax.set_title(title)
            ax.grid(alpha=0.25)
        for ax in axes[-1]:
            ax.set_xlabel('Training iteration')
        save(fig, 'fig_advgrl_diagnostics')

print(f'generated figures in {OUT}')
for path in sorted(OUT.glob('fig_*.png')):
    print(path.name)
