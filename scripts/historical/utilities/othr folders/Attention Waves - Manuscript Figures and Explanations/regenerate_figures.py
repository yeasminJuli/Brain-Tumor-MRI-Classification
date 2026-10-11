"""Regenerate the two manuscript figures from previously extracted original-model weights."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
with (OUT.parent / 'original_attention_visualization/attention_weights_all_test_images.csv').open() as f:
    all_rows = list(csv.DictReader(f))
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
pretty = dict(zip(classes, ['Glioma', 'Meningioma', 'No tumor', 'Pituitary tumor']))
correct = [next(r for r in all_rows if r['true_class'] == c and r['correct'] == 'True') for c in classes]
# One example per observed true/predicted error pair, using alphabetical filename.
errors = sorted((r for r in all_rows if r['correct'] == 'False'), key=lambda r: Path(r['image']).name)
selected_errors = {}
for r in errors:
    selected_errors.setdefault((r['true_class'], r['predicted_class']), r)
incorrect = list(selected_errors.values())


def draw(rows, name):
    assert len(rows) == 4
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    for i, r in enumerate(rows):
        y = .80 - i * .205
        ax = fig.add_axes([.045, y-.012, .14, .16])
        ax.imshow(Image.open(r['image']).convert('L'), cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        ax.text(.5, -.08, f"({chr(97+i)}) {pretty[r['true_class']]}", transform=ax.transAxes, ha='center', fontsize=11)
        ax = fig.add_axes([.235, y, .635, .105])
        w = np.array([float(r[f'weight_{j}']) for j in range(1, 9)])
        assert np.isfinite(w).all() and np.isclose(w.sum(), 1, atol=1e-6)
        im = ax.imshow(w[None, :], cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for j, v in enumerate(w):
            ax.text(j, 0, f'{v:.3f}', ha='center', va='center', fontsize=13,
                    color='white' if v > .55 else '#142536')
        ax.set_xticks(range(8), range(1, 9))
        ax.set_yticks([])
        ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
        ax.grid(which='minor', axis='x', color='white', linewidth=3)
        ax.tick_params(which='minor', bottom=False)
        ax.tick_params(axis='x', labelsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"True: {pretty[r['true_class']]}  |  Predicted: {pretty[r['predicted_class']]}",
                     loc='left', fontsize=12, pad=13)
        if i == 3:
            ax.set_xlabel('Sequence step (feature group)', fontsize=12, labelpad=10)
    cax = fig.add_axes([.91, .27, .018, .50])
    cb = fig.colorbar(im, cax=cax, ticks=[0, .2, .4, .6, .8, 1])
    cb.set_label('Attention weight — darker = greater weight', fontsize=11, labelpad=12)
    fig.text(.055, .062, 'Each strip sums to 1 (before rounding). Equal weighting = 0.125 per step.', fontsize=11)
    fig.text(.055, .037, 'These are contextual feature weights, not tumor locations or segmentation masks.', fontsize=11)
    fig.savefig(OUT / f'{name}.png', dpi=300)
    plt.close(fig)


draw(correct, '01_Correctly_Classified_Attention_Weights')
draw(incorrect, '02_Misclassified_Attention_Weights')
with (OUT / 'Selected_Images_and_Exact_Weights.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['figure', 'panel'] + list(all_rows[0]))
    writer.writeheader()
    for kind, rows in [('correct', correct), ('misclassified', incorrect)]:
        for i, row in enumerate(rows):
            writer.writerow({'figure': kind, 'panel': chr(97+i), **row})
print('Saved two figures and their selected image details to', OUT)
