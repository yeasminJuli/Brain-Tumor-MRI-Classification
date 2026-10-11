"""Sample-only preprocessing audit. No training, test use, or source modifications."""
from pathlib import Path
import csv
import json
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'C:/user/juli/thesis/original_file/othr folders')
DATA = ROOT / 'clean_deduplicated_datasets/Dataset_A/train'
OUT = ROOT / 'Reviewer 2 Comment 7 - Preprocessing Pilot'
CONFIGS = [('Original',1.,8,60),('Clip 0.5',.5,8,60),('Clip 2.0',2.,8,60),
           ('Grid 4x4',1.,4,60),('Grid 16x16',1.,16,60),
           ('Threshold 40',1.,8,40),('Threshold 50',1.,8,50),
           ('Threshold 70',1.,8,70),('Threshold 80',1.,8,80),('No smoothing',1.,8,None)]

def process(gray, clip, grid, threshold):
    normalized = gray / 255.0
    enhanced = cv2.createCLAHE(clipLimit=clip,tileGridSize=(grid,grid)).apply((normalized*255).astype(np.uint8))
    variance = float(cv2.Laplacian(enhanced,cv2.CV_64F).var())
    smooth = threshold is not None and variance < threshold
    if smooth:
        enhanced = cv2.GaussianBlur(enhanced,(3,3),0)
    _, mask = cv2.threshold(enhanced,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=2)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x,y,w,h = cv2.boundingRect(max(contours,key=cv2.contourArea))
        enhanced = enhanced[y:y+h,x:x+w]
    return cv2.resize(enhanced,(256,256)), variance, bool(smooth)

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    raise RuntimeError("Pilot retired: Dataset_A/train contains already-preprocessed inputs. Use a verified raw source for sensitivity analysis.")
    OUT.mkdir(exist_ok=True)
    rng=random.Random(42)
    samples=[]
    for folder in sorted(p for p in DATA.iterdir() if p.is_dir()):
        files=sorted(p for p in folder.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp'])
        samples.extend((folder.name,p) for p in sorted(rng.sample(files,min(25,len(files)))))
    metrics=[];images={};base_vars={}
    for cls,path in samples:
        raw=cv2.imread(str(path))
        if raw is None: raise RuntimeError(f'Unreadable: {path}')
        gray=cv2.resize(cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY),(256,256))
        baseline,var,_=process(gray,1.,8,60)
        base_vars[str(path)]=var
        images[str(path)]={'Input':gray}
        for name,clip,grid,threshold in CONFIGS:
            processed,var,smoothed=process(gray,clip,grid,threshold)
            delta=np.abs(processed.astype(float)-baseline.astype(float))
            metrics.append(dict(image=str(path),class_name=cls,configuration=name,clip_limit=clip,
                                grid=grid,threshold=threshold,laplacian_variance=var,smoothed=smoothed,
                                mean_absolute_pixel_change=float(delta.mean()),changed_pixels=int(np.count_nonzero(delta))))
            images[str(path)][name]=processed
    write_csv(OUT/'Per_Image_Measurements.csv',metrics)
    summary=[]
    for name,*_ in CONFIGS:
        group=[r for r in metrics if r['configuration']==name]
        summary.append(dict(configuration=name,images=len(group),smoothed=sum(r['smoothed'] for r in group),
                            changed_from_original=sum(r['changed_pixels']>0 for r in group),
                            mean_absolute_pixel_change=float(np.mean([r['mean_absolute_pixel_change'] for r in group]))))
    write_csv(OUT/'Configuration_Summary.csv',summary)
    # Deterministic illustrative selection: sampled image nearest the class median baseline variance.
    selected=[]
    for cls in sorted(set(c for c,p in samples)):
        members=[p for c,p in samples if c==cls]
        median=float(np.median([base_vars[str(p)] for p in members]))
        path=min(members,key=lambda p:abs(base_vars[str(p)]-median))
        selected.append((cls,path))
    for filename,columns in [('01_CLAHE_Comparison',['Input','Original','Clip 0.5','Clip 2.0','Grid 4x4','Grid 16x16']),
                             ('02_Smoothing_Comparison',['Original','Threshold 40','Threshold 50','Threshold 70','Threshold 80','No smoothing'])]:
        fig,axes=plt.subplots(4,6,figsize=(13,9))
        for i,(cls,path) in enumerate(selected):
            for j,column in enumerate(columns):
                ax=axes[i,j];ax.imshow(images[str(path)][column],cmap='gray',vmin=0,vmax=255)
                ax.set_xticks([]);ax.set_yticks([])
                if i==0:ax.set_title(column,fontsize=11)
                if j==0:ax.set_ylabel(f'{cls}\nLaplacian var={base_vars[str(path)]:.1f}',fontsize=10)
        fig.tight_layout(rect=(0,.06,1,1))
        fig.text(.03,.025,'Four illustrative sampled training images. Original: clip=1.0, grid=8x8, threshold=60. No model performance evaluated.',fontsize=10)
        fig.savefig(OUT/f'{filename}.png',dpi=220);plt.close(fig)
    write_csv(OUT/'Visual_Example_Names.csv',[dict(class_name=c,image=str(p),baseline_laplacian_variance=base_vars[str(p)]) for c,p in selected])
    values=np.array(list(base_vars.values()))
    info={'source':str(DATA),'sample_size':len(samples),'per_class':25,'seed':42,'opencv':cv2.__version__,
          'sampling':'Seeded random sample within each class; training directory only.',
          'visual_selection':'Nearest to within-sample class median baseline Laplacian variance.',
          'baseline_variance_min_median_max':[float(values.min()),float(np.median(values)),float(values.max())],
          'scope':'Preliminary image-level sensitivity only. No retraining or classification evaluation. Does not establish original selection rationale or optimality.'}
    (OUT/'Pilot_Details.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
