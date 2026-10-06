# Makes the comparison charts in images/ from the saved test predictions.
# Run both training scripts first (they write results/*_predictions.csv).

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, roc_curve, confusion_matrix

# Colors and type
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8984'
GRID = '#e6e5e0'
SYNTHETIC = '#2a78d6'  # blue
REAL = '#eb6834'       # orange
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

if any('Inter' in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams['font.family'] = 'Inter'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.edgecolor': GRID, 'font.size': 15,
})

SIZE = (8, 8)  # 1200 x 1200 px at 150 dpi, square for a LinkedIn feed
DPI = 150

models = {
    'Synthetic churn data': ('results/churn_predictions.csv', SYNTHETIC),
    'Real credit card data': ('results/credit_default_predictions.csv', REAL),
}
results = {}
for name, (path, color) in models.items():
    d = pd.read_csv(path)
    pred = d.y_prob > 0.5
    results[name] = {
        'data': d, 'color': color,
        'Accuracy': accuracy_score(d.y_true, pred),
        'Baseline accuracy': 1 - d.y_true.mean(),
        'ROC AUC': roc_auc_score(d.y_true, d.y_prob),
        'Recall': recall_score(d.y_true, pred),
    }


def header(fig, title, subtitle):
    fig.text(0.07, 0.94, title, fontsize=26, fontweight='bold', color=INK, va='top')
    fig.text(0.07, 0.885, subtitle, fontsize=15, color=INK_2, va='top', linespacing=1.4)


def footer(fig, text):
    fig.text(0.07, 0.035, text, fontsize=11, color=MUTED, va='bottom')


def clean(ax, grid_axis='y'):
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


SOURCE = 'Real data: UCI "Default of Credit Card Clients", Yeh (2009), CC BY 4.0'

# 1. Headline comparison
fig = plt.figure(figsize=SIZE, dpi=DPI)
fig.text(0.07, 0.94, 'Same neural network,\nsynthetic vs real data', fontsize=26, fontweight='bold', color=INK,
         va='top', linespacing=1.15)
fig.text(0.07, 0.80, 'Test-set scores of a small Keras ANN, trained once on synthetic\n'
         'bank-churn data and once on 30,000 real bank customers.',
         fontsize=15, color=INK_2, va='top', linespacing=1.4)
ax = fig.add_axes([0.07, 0.16, 0.86, 0.48])
metrics = ['Accuracy', 'ROC AUC', 'Recall']
labels = ['Accuracy', 'ROC AUC', 'Recall on\npositive class']
width = 0.34
for i, (name, r) in enumerate(results.items()):
    xs = [m + (i - 0.5) * (width + 0.02) for m in range(len(metrics))]
    vals = [r[m] for m in metrics]
    ax.bar(xs, vals, width=width, color=r['color'], label=name, zorder=2)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.015, f'{v:.2f}', ha='center', va='bottom', fontsize=15, color=INK, fontweight='medium')
ax.set_xticks(range(len(metrics)), labels, fontsize=15, color=INK)
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
clean(ax)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=14,
          handlelength=1, handleheight=1, borderaxespad=0)
footer(fig, SOURCE)
fig.savefig('images/1_synthetic_vs_real.png')
plt.close(fig)

# 2. ROC curves
fig = plt.figure(figsize=SIZE, dpi=DPI)
header(fig, 'Real data is harder to separate',
       'ROC curves on the held-out test sets. The closer a curve hugs\nthe top-left corner, the better the model ranks customers.')
ax = fig.add_axes([0.12, 0.17, 0.81, 0.58])
ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.5, linestyle=(0, (4, 4)), zorder=1)
ax.text(0.62, 0.55, 'random guess', color=MUTED, fontsize=12, rotation=45, ha='center', va='center',
        rotation_mode='anchor', transform_rotates_text=True)
for name, r in results.items():
    fpr, tpr, _ = roc_curve(r['data'].y_true, r['data'].y_prob)
    ax.plot(fpr, tpr, color=r['color'], linewidth=2.5, label=f"{name} (AUC {r['ROC AUC']:.2f})", zorder=2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.01)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
clean(ax, grid_axis='both')
ax.spines['left'].set_visible(True)
ax.spines['left'].set_color(GRID)
ax.legend(loc='lower right', frameon=False, fontsize=14, handlelength=1.5)
footer(fig, SOURCE)
fig.savefig('images/2_roc_curves.png')
plt.close(fig)

# 3. Confusion matrix on the real data
r = results['Real credit card data']
cm = confusion_matrix(r['data'].y_true, r['data'].y_prob > 0.5)
fig = plt.figure(figsize=SIZE, dpi=DPI)
header(fig, 'Where the real-data model goes wrong',
       f"6,000 test customers. It's right {r['Accuracy']:.0%} of the time, but catches\n"
       f"only {r['Recall']:.0%} of the customers who actually defaulted.")
ax = fig.add_axes([0.24, 0.14, 0.66, 0.60])
cells = [[cm[0, 0], cm[0, 1]], [cm[1, 0], cm[1, 1]]]
names = [['Correctly cleared', 'False alarm'], ['Missed default', 'Caught default']]
vmax = cm.max()
gap = 0.03
for i in range(2):
    for j in range(2):
        v = cells[i][j]
        shade = BLUES[min(len(BLUES) - 1, round((len(BLUES) - 1) * (0.15 + 0.85 * v / vmax)))]
        ax.add_patch(plt.Rectangle((j + gap, 1 - i + gap), 1 - 2 * gap, 1 - 2 * gap, color=shade,
                                   joinstyle='round'))
        dark = BLUES.index(shade) >= 3
        ax.text(j + 0.5, 1 - i + 0.56, f'{v:,}', ha='center', va='center', fontsize=30, fontweight='bold',
                color='#ffffff' if dark else INK)
        ax.text(j + 0.5, 1 - i + 0.36, names[i][j], ha='center', va='center', fontsize=14,
                color='#ffffff' if dark else INK_2)
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5], ['Predicted:\nno default', 'Predicted:\ndefault'], fontsize=14, color=INK)
ax.set_yticks([1.5, 0.5], ['Actual:\nno default', 'Actual:\ndefault'], fontsize=14, color=INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
footer(fig, SOURCE)
fig.savefig('images/3_confusion_matrix.png')
plt.close(fig)

for name, r in results.items():
    print(f"{name}: accuracy {r['Accuracy']:.3f} (baseline {r['Baseline accuracy']:.3f}), "
          f"ROC AUC {r['ROC AUC']:.3f}, recall {r['Recall']:.3f}")
