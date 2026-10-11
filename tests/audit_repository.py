"""Offline packaging and interface audit; does not retrain historical experiments."""
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    checks = {}
    failures = []
    records = list(csv.DictReader((ROOT/'docs/source_index.csv').open(encoding='utf-8')))
    for row in records:
        p = ROOT/row['package_path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            failures.append('Source hash: '+row['package_path'])
    checks['source_hash_records'] = len(records)
    tracked = subprocess.check_output(['git','ls-files','-z'], cwd=ROOT).decode().split('\0')
    tracked = [ROOT/p for p in tracked if p]
    code_files = [p for p in tracked if p.suffix == '.py']
    for p in code_files:
        try: ast.parse(p.read_text(encoding='utf-8-sig'))
        except Exception as e: failures.append('Syntax: '+str(p.relative_to(ROOT))+': '+str(e))
    checks['tracked_python_files_parsed'] = len(code_files)
    links = 0
    for p in [ROOT/'README.md', *sorted((ROOT/'docs').glob('*.md'))]:
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if target.startswith(('http:', 'https:', '#', 'mailto:')): continue
            target = unquote(target.split('#')[0].strip('<>'))
            if not (p.parent/target).exists(): failures.append(f'Broken link in {p.name}: {target}')
            links += 1
    checks['relative_document_links_checked'] = links
    pngs = [p for p in tracked if p.suffix.lower()=='.png']
    for p in pngs:
        try:
            with Image.open(p) as im: im.verify()
        except Exception: failures.append('Unreadable PNG: '+str(p.relative_to(ROOT)))
    checks['png_files_verified'] = len(pngs)
    forbidden = [p for p in tracked if p.suffix.lower() in {'.pth','.pt','.ckpt','.zip','.jpg','.jpeg'} or p.stat().st_size > 50_000_000]
    failures += ['Unexpected public binary: '+str(p.relative_to(ROOT)) for p in forbidden]
    # Report filenames only, never potential secret contents.
    secret = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    for p in tracked:
        if p.suffix.lower() in {'.py','.md','.json','.txt','.yml','.yaml','.csv'} and p.stat().st_size<5_000_000:
            if secret.search(p.read_text(encoding='utf-8-sig',errors='replace')):
                failures.append('Potential credential: '+str(p.relative_to(ROOT)))
    metrics_checks=[]
    for metrics_file in sorted((ROOT/'results').rglob('overall_metrics.csv')):
        cm_path = metrics_file.parent/'confusion_matrix.csv'
        if not cm_path.exists(): cm_path = metrics_file.parent/'confusion_matrix_counts.csv'
        if not cm_path.exists(): continue
        rows=list(csv.reader(cm_path.open(encoding='utf-8-sig')))
        cm=np.array([[float(v) for v in row[1:]] for row in rows[1:]])
        if cm.shape!=(4,4) or cm.sum()<=0: continue
        metrics=list(csv.DictReader(metrics_file.open(encoding='utf-8-sig')))[0]
        if 'accuracy' not in metrics: continue
        actual=float(cm.trace()/cm.sum())
        if not np.isclose(actual,float(metrics['accuracy']),atol=1e-8): failures.append('Accuracy mismatch: '+str(metrics_file.relative_to(ROOT)))
        per=metrics_file.parent/'per_class_metrics.csv'
        if per.exists():
            class_rows=list(csv.DictReader(per.open(encoding='utf-8-sig')))
            for i,row in enumerate(class_rows):
                if i>=4: break
                precision=cm[i,i]/cm[:,i].sum() if cm[:,i].sum() else 0
                recall=cm[i,i]/cm[i].sum() if cm[i].sum() else 0
                f1=2*precision*recall/(precision+recall) if precision+recall else 0
                for key,value in [('precision',precision),('recall',recall),('f1-score',f1),('support',cm[i].sum())]:
                    if key in row and not np.isclose(float(row[key]),value,atol=1e-8): failures.append(f'Class metric mismatch: {per.relative_to(ROOT)} {key}')
        metrics_checks.append({'file':str(metrics_file.relative_to(ROOT)).replace('\\','/'),'test_images':int(cm.sum()),'recomputed_accuracy':actual})
    checks['confusion_matrix_reconciliations'] = metrics_checks
    import torch
    from src.models.proposed import HybridCNNBiLSTMAttention
    torch.set_num_threads(2)
    model=HybridCNNBiLSTMAttention().eval()
    with torch.inference_mode():
        logits,attention=model(torch.zeros(1,3,256,256))
    assert logits.shape==(1,4) and attention.shape==(1,8,1)
    assert torch.isfinite(logits).all() and torch.allclose(attention.sum(1),torch.ones(1,1))
    checks['model_parameters']=sum(p.numel() for p in model.parameters())
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp); inp=folder/'raw'/'Healthy'; inp.mkdir(parents=True)
        image=np.random.default_rng(42).integers(0,256,(90,120,3),dtype=np.uint8)
        assert cv2.imwrite(str(inp/'synthetic.png'),image)
        cli=ROOT/'scripts/prepare_data.py'
        def run(*args): return subprocess.run([sys.executable,'-B',str(cli),*map(str,args)],capture_output=True,text=True)
        prep=folder/'processed'
        result=run('preprocess','--input',folder/'raw','--output',prep)
        assert result.returncode==0,result.stderr
        saved=list(prep.glob('*/*.png')); assert len(saved)==1
        from src.preprocessing.pipeline import preprocess_image
        assert np.array_equal(cv2.imread(str(saved[0]),0),preprocess_image(str(inp/'synthetic.png')))
        for name in ['aug1','aug2']:
            result=run('augment','--input',prep,'--output',folder/name,'--copies',4,'--seed',42)
            assert result.returncode==0,result.stderr
        files=sorted((folder/'aug1').glob('*/*.png')); assert len(files)==5
        for f in files: assert f.read_bytes()==(folder/'aug2'/f.relative_to(folder/'aug1')).read_bytes()
        assert run('preprocess','--input',folder/'raw','--output',prep).returncode!=0
    checks['synthetic_preprocessing_and_seeded_augmentation']='passed; existing outputs are refused'
    checks['environment']={'python':sys.version.split()[0],'torch':torch.__version__,'numpy':np.__version__,'opencv':cv2.__version__}
    checks['failures']=failures
    (ROOT/'docs/audit_results.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
