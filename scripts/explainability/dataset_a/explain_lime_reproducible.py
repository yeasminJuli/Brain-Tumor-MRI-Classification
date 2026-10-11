"""Regenerate LIME for the original single-run model with explicit, recorded settings.

Default: four correct examples plus every test-set error. --all: all test images.
No training is performed. Old explanations and checkpoint files remain untouched.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import ast
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import datasets, transforms
from lime import lime_image
from sklearn.linear_model import Ridge
from skimage.segmentation import slic, mark_boundaries

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'othr folders'
CHECKPOINT = ROOT / 'training_outputs_proposed_model_orgnl/training_outputs/hybrid_cnn_bilstm_attention_final.pth'
TEST = ROOT / 'preprocessed_testt'
CLASS_ORDER = ['glioma', 'meningioma', 'notumor', 'pituitary']
DISPLAY = {'glioma':'Glioma', 'meningioma':'Meningioma', 'notumor':'No tumor', 'pituitary':'Pituitary tumor'}


def dump_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def model_from_source():
    source = Path(__file__).resolve().parents[3] / 'src' / 'models' / 'proposed.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    names = {'ResidualBlock','CNNFeatureExtractor','AdditiveAttention','HybridCNNBiLSTMAttention'}
    classes = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names], type_ignores=[])
    namespace = {'torch':torch,'nn':nn,'NUM_CLASSES':4}
    exec(compile(classes, str(source), 'exec'), namespace)
    return namespace['HybridCNNBiLSTMAttention'](), source


def panel(records, path):
    fig, axes = plt.subplots(len(records), 2, figsize=(8, 3.25*len(records)), squeeze=False)
    for i, r in enumerate(records):
        with np.load(r['_arrays']) as data:
            original = data['image']; mask = data['positive_mask']
        axes[i,0].imshow(original)
        axes[i,1].imshow(mark_boundaries(original, mask, color=(1,1,0), mode='thick'))
        axes[i,0].set_title(f"({chr(97+i)}) True: {DISPLAY[r['true_class']]}", fontsize=12)
        axes[i,1].set_title(f"Predicted: {DISPLAY[r['predicted_class']]}\nSLIC regions: {r['actual_superpixels']}; highlighted: {r['highlighted_regions']}", fontsize=11)
        axes[i,0].set_xlabel(r['filename'], fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.text(.06,.012,'Yellow boundaries: selected positive LIME regions for the predicted class; not tumor boundaries.', fontsize=9)
    fig.savefig(path, dpi=240); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Explain all test images instead of the review examples.')
    parser.add_argument('--num-samples', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--correct-per-class', type=int, default=1)
    parser.add_argument('--correct-only', action='store_true')
    args = parser.parse_args()
    if args.num_samples < 2: raise ValueError('At least two perturbation samples required')
    if args.correct_per_class < 1: raise ValueError('At least one correct example per class required')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = HERE/'outputs'/'LIME - Reproducible Figures'/stamp
    out.mkdir(parents=True, exist_ok=False)
    detail = out/'per_image'; detail.mkdir()
    torch.set_num_threads(1); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, source = model_from_source()
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.eval().to(device)
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize((256,256)),
        transforms.ToTensor(), transforms.Normalize((.5,)*3,(.5,)*3)])
    ds = datasets.ImageFolder(TEST, transform=transform)
    if ds.classes != CLASS_ORDER: raise RuntimeError(f'Unexpected class order: {ds.classes}')
    if 'class_names' in checkpoint and list(checkpoint['class_names']) != CLASS_ORDER:
        raise RuntimeError('Checkpoint class order differs from training ImageFolder order')

    def predict(images):
        batch = torch.stack([transform(Image.fromarray(np.asarray(im,dtype=np.uint8))) for im in images]).to(device)
        with torch.inference_mode(): logits, _ = model(batch)
        probs = logits.softmax(1).cpu().numpy()
        if not np.isfinite(probs).all(): raise RuntimeError('Non-finite predictions')
        return probs

    print('Output:',out,flush=True)
    predictions=[]
    with torch.inference_mode():
        for images, labels in torch.utils.data.DataLoader(ds,batch_size=10,shuffle=False,num_workers=0):
            logits,_ = model(images.to(device)); probs=logits.softmax(1).cpu().numpy()
            for p,label in zip(probs,labels.tolist()):
                path=ds.samples[len(predictions)][0]; index=int(p.argmax())
                predictions.append(dict(image=path,filename=Path(path).name,true_class=CLASS_ORDER[label],
                    predicted_class=CLASS_ORDER[index],predicted_index=index,correct=index==label,
                    probability=float(p[index]),probabilities=p.tolist()))
    dump_csv(out/'All_Test_Predictions.csv',[{**r,'probabilities':json.dumps(r['probabilities'])} for r in predictions])
    correct=[next(r for r in predictions if r['true_class']==c and r['correct']) for c in CLASS_ORDER]
    candidates=[r for c in CLASS_ORDER for r in [p for p in predictions if p['true_class']==c and p['correct']][:args.correct_per_class]]
    errors=[r for r in predictions if not r['correct']]
    selected = predictions if args.all else candidates+([] if args.correct_only else errors)
    # Start at zero: LIME perturbs feature IDs 0 through K-1.
    slic_settings=dict(n_segments=50,compactness=10.,sigma=1.,max_num_iter=10,
        convert2lab=True,enforce_connectivity=True,min_size_factor=.5,max_size_factor=3.,
        slic_zero=False,start_label=0,channel_axis=-1,spacing=None,mask=None)
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        python=platform.python_version(),platform=platform.platform(),
        packages={n:metadata.version(n) for n in ['lime','scikit-image','numpy','torch','torchvision','scikit-learn','Pillow','matplotlib']},
        checkpoint=str(CHECKPOINT),checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        model_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),class_order=CLASS_ORDER,
        class_order_source='Verified training ImageFolder alphabetical order; checkpoint has no class metadata unless checked above.',
        input='Existing preprocessed test image, RGB container; predictor replicates grayscale to 3 channels, resizes to 256x256, normalizes mean/std 0.5.',
        slic=slic_settings,kernel_width=.25,kernel='sqrt(exp(-distance^2 / kernel_width^2))',distance_metric='cosine',
        num_samples=args.num_samples,prediction_batch_size=10,hide_color=0,
        surrogate='Ridge(alpha=1, fit_intercept=True, solver=cholesky); all actual superpixels used as regressors',
        explained_class='Predicted class only: labels=(predicted_index,), top_labels=None',
        positive_only=True,hide_rest=False,max_displayed_regions=10,min_weight=0.,
        base_seed=args.seed,seed_rule='(base_seed + first 8 hex digits of SHA256(relative image path)) modulo 2^32',
        deterministic_pytorch=True,tf32=False,selection='All test images' if args.all else f'First {args.correct_per_class} correct filenames per class in ImageFolder order'+('; correct examples only' if args.correct_only else '; plus every misclassification')+'; no selection by explanation quality.',
        total_test_images=len(predictions),test_errors=len(errors),test_accuracy=float(np.mean([r['correct'] for r in predictions])),
        explanation_images=len(selected),historical_note='New regenerated explanations; fixed seeds, explicit segmentation labels and corrected display class order. Not exact reproduction of the unseeded original figures.')
    (out/'Run_Configuration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    shutil.copy2(__file__,out/'Script_Snapshot.py')
    shutil.copy2(source,out/'Model_Source_Snapshot.txt')
    results=[]
    for number,r in enumerate(selected,1):
        rel=Path(r['image']).relative_to(TEST).as_posix()
        seed=(args.seed+int(hashlib.sha256(rel.encode()).hexdigest()[:8],16)) % (2**32)
        with Image.open(r['image']) as img: image=np.asarray(img.convert('RGB'))
        segments=slic(image,**slic_settings)
        k=len(np.unique(segments))
        if not np.array_equal(np.unique(segments),np.arange(k)): raise RuntimeError('Noncontiguous superpixel labels')
        p=predict([image])[0]
        if int(p.argmax()) != r['predicted_index']: raise RuntimeError('Prediction differs between evaluation and LIME wrapper')
        def explain():
            explainer=lime_image.LimeImageExplainer(kernel_width=.25,feature_selection='none',random_state=seed)
            return explainer.explain_instance(image,predict,labels=(r['predicted_index'],),top_labels=None,
                hide_color=0,num_features=k,num_samples=args.num_samples,batch_size=10,
                segmentation_fn=lambda _:segments,distance_metric='cosine',random_seed=seed,
                model_regressor=Ridge(alpha=1.,fit_intercept=True,solver='cholesky'))
        explanation=explain()
        _,mask=explanation.get_image_and_mask(r['predicted_index'],positive_only=True,negative_only=False,
            hide_rest=False,num_features=10,min_weight=0.)
        pairs=explanation.local_exp[r['predicted_index']]
        weights=np.zeros(k)
        for idx,w in pairs: weights[idx]=w
        if not np.isfinite(weights).all(): raise RuntimeError('Invalid explanation coefficients')
        repeat_result=None
        if number==1:
            repeated=explain()
            repeated_weights=np.zeros(k)
            for idx,w in repeated.local_exp[r['predicted_index']]: repeated_weights[idx]=w
            _,repeat_mask=repeated.get_image_and_mask(r['predicted_index'],positive_only=True,hide_rest=False,num_features=10,min_weight=0.)
            repeat_result=dict(max_coefficient_difference=float(np.max(np.abs(weights-repeated_weights))),same_mask=bool(np.array_equal(mask,repeat_mask)))
            if not np.allclose(weights,repeated_weights,atol=1e-8,rtol=1e-6) or not repeat_result['same_mask']:
                raise RuntimeError('Seeded repeat check failed')
            (out/'Seeded_Repeat_Check.json').write_text(json.dumps(repeat_result,indent=2))
        stem=f"{r['true_class']}__{Path(r['image']).stem}"
        arrays=detail/f'{stem}.npz'
        np.savez_compressed(arrays,image=image,segments=segments,positive_mask=mask,coefficients=weights,probabilities=p)
        record={**r,'probability':float(p[r['predicted_index']]),'image_sha256':hashlib.sha256(Path(r['image']).read_bytes()).hexdigest(),
            'seed':seed,'actual_superpixels':k,'highlighted_regions':int(len(np.unique(segments[mask>0]))),
            'weighted_local_R2':float(explanation.score),'surrogate_prediction':float(np.asarray(explanation.local_pred).item()),
            'intercept':float(explanation.intercept[r['predicted_index']]),'_arrays':str(arrays)}
        (detail/f'{stem}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        panel([record],detail/f'{stem}.png')
        results.append(record)
        dump_csv(out/'Explanation_Summary.csv',[{key:json.dumps(value) if isinstance(value,list) else value for key,value in rec.items()} for rec in results])
        print(f"[{number}/{len(selected)}] {rel}: predicted {r['predicted_class']}, segments={k}, highlighted={record['highlighted_regions']}, local R2={explanation.score:.3f}",flush=True)
    mapping={r['image']:r for r in results}
    panel([mapping[r['image']] for r in correct],out/'01_Correctly_Classified_LIME.png')
    included_errors=[r for r in errors if r['image'] in mapping]
    for start in range(0,len(included_errors),4):
        panel([mapping[r['image']] for r in included_errors[start:start+4]],out/f'02_Misclassified_LIME_{start//4+1}.png')
    if args.correct_per_class > 1:
        for cls in CLASS_ORDER:
            panel([mapping[r['image']] for r in candidates if r['true_class']==cls],out/f'Candidates_{cls}.png')
    counts=[r['actual_superpixels'] for r in results]
    low=sum(r['weighted_local_R2']<.5 for r in results)
    report=f'''REGENERATED LIME EXPLANATIONS - ORIGINAL SINGLE-RUN CHECKPOINT

This run generated {len(results)} explanations. Test-set prediction check: {len(errors)} errors among {len(predictions)} images ({config['test_accuracy']*100:.2f}% accuracy). No retraining was performed.

MANUSCRIPT METHODS
LIME explanations were regenerated using SLIC segmentation with a target of 50 superpixels, compactness 10, and Gaussian smoothing sigma 1. Segments were assigned contiguous zero-based labels. The actual number of superpixels ranged from {min(counts)} to {max(counts)} across the explained images and is recorded separately for each image. For each input, {args.num_samples} binary superpixel perturbation samples were generated, including the unperturbed input, with masked regions replaced by black pixels before model normalization. Cosine distance and an exponential kernel with width 0.25 weighted the perturbed samples. A weighted ridge-regression surrogate with regularization alpha=1 was fitted using all actual superpixels. Explanations targeted the model's predicted class and displayed up to 10 positive-coefficient superpixels, with the remaining image visible. Yellow outlines indicate the selected regions, not verified tumor boundaries. Perturbation seeds were deterministically derived from base seed {args.seed} and the relative image path; the exact seeds, software versions, and segmentation settings are included in Run_Configuration.json and per-image records. Inference used {config['gpu'] or 'CPU'}.

FIGURE CAPTION - CORRECT PREDICTIONS
Original preprocessed test inputs and regenerated LIME visualizations for correctly classified examples from the four classes. Yellow boundaries enclose up to ten positively contributing superpixels for the predicted class. Ground-truth and predicted labels are shown explicitly. Actual SLIC and highlighted-region counts are reported per example. These are local surrogate-model explanations, not tumor segmentation masks.

FIGURE CAPTION - ERRORS
Original preprocessed test inputs and regenerated LIME visualizations for misclassified examples. Each explanation targets the predicted class, which may differ from the ground-truth label. Yellow boundaries show selected positively contributing superpixels. The visualizations do not alone establish why a prediction was incorrect.

INTERPRETATION AND QUALITY
The weighted local R2 describes how well the surrogate fits model probabilities over the sampled perturbations; it is not clinical validity or segmentation accuracy. {low}/{len(results)} explanations have weighted local R2 below 0.5 (a descriptive flag, not a validated cutoff). Inspect Explanation_Summary.csv before making fidelity claims. A fixed-seed repeat on the first image verified matching selected masks and numerically matching coefficients. This checks reproducibility under these settings, not stability across different random seeds or software/hardware environments.

WHAT CHANGED
Corrected output index labels to glioma, meningioma, notumor, pituitary. Explicit kernel width and settings. Fixed per-image seeds. Zero-based SLIC labels compatible with LIME's binary perturbation indexing; the old script left start_label to the segmentation library default. These are newly generated outputs and should replace the old unseeded figures if adopted. Do not describe these new seeds as historical settings. Do not change historical training hardware unless that training was rerun or records establish a correction.

FILES
01_Correctly_Classified_LIME.png: four correct examples.
02_Misclassified_LIME_*.png: all test errors grouped in panels.
per_image/: individual figures, actual segment maps, masks, coefficients and provenance.
Explanation_Summary.csv: actual counts and fidelity for all explained images.
Run_Configuration.json: complete explicit settings, versions, hardware and checkpoint hash.
All_Test_Predictions.csv: class-mapping and prediction verification.
Seeded_Repeat_Check.json: reproducibility check.

SCOPE
These examples document LIME reproducibility. They do not establish anatomical accuracy or quantitative localization performance.
'''
    (out/'00_READ_ME_Methods_Captions_and_Quality.txt').write_text(report,encoding='utf-8')
    print('COMPLETE:',out,flush=True)


if __name__=='__main__':main()
