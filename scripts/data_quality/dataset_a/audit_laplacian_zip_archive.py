"""Read-only original Dataset A audit directly from its ZIP archive."""
from pathlib import Path
import csv,json,zipfile,hashlib
import cv2
import numpy as np
ROOT=Path(r'C:/user/juli/thesis/original_file/othr folders/Original_data.zip')
OUT=Path(r'C:/user/juli/thesis/original_file/othr folders/Original Dataset A ZIP Audit')

def main():
 OUT.mkdir(exist_ok=True)
 rows=[];errors=[]
 archive=zipfile.ZipFile(ROOT)
 paths=sorted(n for n in archive.namelist() if n.lower().endswith(('.jpg','.jpeg','.png','.bmp')))
 for i,p in enumerate(paths):
  img=cv2.imdecode(np.frombuffer(archive.read(p),dtype=np.uint8),cv2.IMREAD_COLOR)
  if img is None:
   errors.append(str(p));continue
  gray=cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(256,256))
  enhanced=cv2.createCLAHE(clipLimit=1.,tileGridSize=(8,8)).apply(((gray/255.)*255).astype(np.uint8))
  variance=float(cv2.Laplacian(enhanced,cv2.CV_64F).var())
  rel=Path(p)
  rows.append(dict(image=str(p),split=rel.parts[-3],class_name=rel.parts[-2],width=img.shape[1],height=img.shape[0],laplacian_variance=variance,apply_smoothing_60=variance<60))
  if (i+1)%2000==0:print(f'Checked {i+1}/{len(paths)}',flush=True)
 with (OUT/'All_Raw_Dataset_A_Images.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 selected=[r for r in rows if r['apply_smoothing_60']]
 with (OUT/'Dataset_A_Images_Below_60.csv').open('w',newline='') as f:
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
 with (OUT/'Dataset_A_Counts_By_Split_And_Class.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 info=dict(archive_sha256=hashlib.sha256(ROOT.read_bytes()).hexdigest(),source=str(ROOT),dataset='Original Dataset A ZIP, 7023 images',images=len(rows),unreadable=errors,opencv=cv2.__version__,procedure='BGR grayscale; 256x256 INTER_LINEAR resize; existing normalization round trip; 8-bit CLAHE clip=1 grid=8x8; CV_64F Laplacian defaults; variance <60 BEFORE smoothing/Otsu/cropping',summary=summary)
 (OUT/'Audit_Details.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2));print('Unreadable:',len(errors))

if __name__=='__main__':main()
