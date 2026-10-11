"""Standalone attention strips for the original checkpoint; no training or Grad-CAM.

Loads only model class definitions via AST to avoid training-module import side effects.
Preserves the original direct view reshaping: steps are feature groups, not image columns.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(r'C:/user/juli/thesis/original_file/othr folders')
CHECKPOINT = ROOT / 'training_outputs_proposed_model_orgnl/training_outputs/hybrid_cnn_bilstm_attention_final.pth'
TEST = ROOT / 'preprocessed_testt'
OUT = ROOT / 'original_attention_visualization'
SOURCE = Path(__file__).with_name('train_model_withCustomized_cnn.py')


def plot_strips(rows, filename, title, vmax):
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 1.2 * len(rows) + 2.2), squeeze=False)
    for ax, row in zip(axes[:, 0], rows):
        weights = np.array([row[f'weight_{i}'] for i in range(1, 9)])
        im = ax.imshow(weights[None, :], cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
        for j, value in enumerate(weights):
            ax.text(j, 0, f'{value:.3f}', ha='center', va='center',
                    color='white' if value / vmax > .55 else '#142536', fontsize=12)
        ax.set_xticks(range(8), [str(i) for i in range(1, 9)])
        ax.set_yticks([])
        ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
        ax.grid(which='minor', axis='x', color='white', linewidth=3)
        ax.tick_params(which='minor', bottom=False)
        ax.set_title(f"{Path(row['image']).name}  |  True: {row['true_class']}  |  Predicted: {row['predicted_class']}",
                     loc='left', fontsize=11, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1, 0].set_xlabel('Sequence step (feature group)', fontsize=11)
    fig.suptitle(title, fontsize=17, fontweight='bold', y=.98)
    fig.subplots_adjust(left=.06, right=.85, top=.87, bottom=.19, hspace=.95)
    cax = fig.add_axes([.89, .26, .022, .49])
    fig.colorbar(im, cax=cax).set_label('Attention weight — darker = greater weight', fontsize=10)
    fig.text(.06, .045, 'Each strip sums to 1 (before rounding). Equal weighting = 0.125 per step.\n'
             'These are contextual feature weights, not tumor locations or segmentation masks.', fontsize=10)
    fig.savefig(OUT / f'{filename}.png', dpi=220)
    fig.savefig(OUT / f'{filename}.svg')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    names = {'ResidualBlock', 'CNNFeatureExtractor', 'AdditiveAttention', 'HybridCNNBiLSTMAttention'}
    parsed = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    module = ast.Module(body=[n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name in names], type_ignores=[])
    namespace = {'torch': torch, 'nn': nn, 'NUM_CLASSES': 4}
    exec(compile(module, str(SOURCE), 'exec'), namespace)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model = namespace['HybridCNNBiLSTMAttention']().to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    dataset = datasets.ImageFolder(TEST, transform=transform)
    rows = []
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0):
            logits, alpha = model(images.to(device))
            weights = alpha[:, :, 0].cpu().numpy()
            assert weights.shape[1] == 8 and np.isfinite(weights).all()
            assert (weights >= 0).all() and np.allclose(weights.sum(1), 1, atol=1e-6)
            probs = logits.softmax(1).cpu().numpy()
            for i, label in enumerate(labels.tolist()):
                pred = int(probs[i].argmax())
                row = {'image': dataset.samples[len(rows)][0], 'true_class': dataset.classes[label],
                       'predicted_class': dataset.classes[pred], 'correct': pred == label,
                       'predicted_probability': float(probs[i, pred])}
                row.update({f'weight_{j+1}': float(w) for j, w in enumerate(weights[i])})
                rows.append(row)
            if len(rows) % 160 == 0:
                print(f'Processed {len(rows)}/{len(dataset)}', flush=True)
    with (OUT / 'attention_weights_all_test_images.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    vmax = max(r[f'weight_{i}'] for r in rows for i in range(1, 9))
    representatives = [next(r for r in rows if r['true_class'] == c and r['correct']) for c in dataset.classes]
    plot_strips(representatives, 'attention_four_classes', 'Original model: attention weights', vmax)
    errors = [r for r in rows if not r['correct']]
    for start in range(0, len(errors), 6):
        plot_strips(errors[start:start+6], f'attention_errors_{start//6+1}', 'Attention weights: misclassified test images', vmax)
    metadata = {'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        'model_source': str(SOURCE), 'test_directory': str(TEST), 'classes': dataset.classes,
        'class_mapping_note': 'Alphabetical ImageFolder order; checkpoint contains no class_names metadata.',
        'images': len(rows), 'errors': len(errors), 'accuracy': sum(r['correct'] for r in rows)/len(rows),
        'device': str(device), 'checkpoint_epoch': checkpoint.get('epoch'),
        'color_scale': [0, vmax], 'selection': 'First correctly classified file in alphabetical order per class; all errors.',
        'interpretation': 'Original direct NCHW view creates eight channel groups (32 channels each), not spatial columns. Weights are contextual and not class-specific tumor localization.',
        'validation': 'Strict checkpoint load; all attention weights finite, nonnegative and sum to one.'}
    (OUT / 'run_details.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
