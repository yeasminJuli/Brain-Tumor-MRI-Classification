"""Read-only full raw Dataset B audit; this is NOT original Dataset A."""
from pathlib import Path
import csv,json
import cv2
import numpy as np
ROOT=Path(r'C:/user/juli/thesis/Dataset_B/Dataset/Brain Tumor MRI Dataset (G,M,P)_data')
OUT=Path(r'C:/user/juli/thesis/original_file/othr folders/Full Raw Image Audit')

def main():
 OUT.mkdir(exist_ok=True)
 rows=[];errors=[]
 paths=sorted(p for p in ROOT.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp'])
 for i,p in enumerate(paths):
  img=cv2.imread(str(p))
  if img is None:
   errors.append(str(p));continue
  gray=cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(256,256))
  enhanced=cv2.createCLAHE(clipLimit=1.,tileGridSize=(8,8)).apply(((gray/255.)*255).astype(np.uint8))
  variance=float(cv2.Laplacian(enhanced,cv2.CV_64F).var())
  rel=p.relative_to(ROOT)
  rows.append(dict(image=str(p),split=rel.parts[0],class_name=rel.parts[1],width=img.shape[1],height=img.shape[0],laplacian_variance=variance,apply_smoothing_60=variance<60))
  if (i+1)%2000==0:print(f'Checked {i+1}/{len(paths)}',flush=True)
 with (OUT/'All_Raw_Dataset_B_Images.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 selected=[r for r in rows if r['apply_smoothing_60']]
 with (OUT/'Dataset_B_Images_Below_60.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(selected)
 summary=[]
 for split in ['Train','Test','ALL']:
  for cls in ['glioma','meningioma','notumor','pituitary','ALL']:
   group=[r for r in rows if (split=='ALL' or r['split']==split) and (cls=='ALL' or r['class_name']==cls)]
   if not group:continue
   v=np.array([r['laplacian_variance'] for r in group])
   item=dict(split=split,class_name=cls,images=len(v),min_variance=float(v.min()),median_variance=float(np.median(v)))
   item.update({f'below_{t}':int((v<t).sum()) for t in [40,50,60,70,80]})
   summary.append(item)
 with (OUT/'Dataset_B_Counts_By_Split_And_Class.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 info=dict(source=str(ROOT),dataset='Raw Dataset B; NOT original 7023-image Dataset A',images=len(rows),unreadable=errors,opencv=cv2.__version__,procedure='BGR grayscale; 256x256 INTER_LINEAR resize; existing normalization round trip; 8-bit CLAHE clip=1 grid=8x8; CV_64F Laplacian defaults; variance <60 BEFORE smoothing/Otsu/cropping',summary=summary)
 (OUT/'Audit_Details.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2));print('Unreadable:',len(errors))

if __name__=='__main__':main()
