import os, json, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

DATA_DIR = (r"C:\user\juli\thesis\original_file\othr folders" r"\PREPROCESSED_TRAINN_AUGMENTED")
FIXED_FOLD_SOURCE = (r"C:\user\juli\thesis\original_file\othr folders" r"\CV_RESULTS_PROPOSED_MODEL")
FIXED_FOLDS_FILE = os.path.join(FIXED_FOLD_SOURCE, "FIXED_5_FOLD_SPLITS.npz")
SAVE_ROOT = (r"C:\user\juli\thesis\original_file\othr folders" r"\CV_RESULTS_EFFICIENTNET_B3_SAFE_RERUN")
os.makedirs(SAVE_ROOT, exist_ok=True)

SEED=42; N_SPLITS=5; NUM_CLASSES=4; INPUT_SIZE=(256,256); BATCH_SIZE=16; LEARNING_RATE=1e-4; NUM_EPOCHS=50; EARLY_STOPPING_PATIENCE=5; MIN_DELTA=0.0; NUM_WORKERS=0; PRETRAINED=True
CLASS_NAMES=["glioma","meningioma","notumor","pituitary"]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP=DEVICE.type=="cuda"

print("="*80); print("EFFICIENTNET-B3 - 5-FOLD STRATIFIED CROSS-VALIDATION"); print("="*80)
for k,v in [("Device",DEVICE),("Dataset",DATA_DIR),("Fixed folds",FIXED_FOLDS_FILE),("Save folder",SAVE_ROOT),("Folds / Epochs",f"{N_SPLITS} / {NUM_EPOCHS}"),("Batch / LR",f"{BATCH_SIZE} / {LEARNING_RATE}"),("Workers",NUM_WORKERS),("Pretrained",PRETRAINED),("Early stopping",EARLY_STOPPING_PATIENCE)]: print(f"{k:18}: {v}")
print("="*80)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
set_seed(SEED)

transform=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.Resize(INPUT_SIZE),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
dataset=datasets.ImageFolder(DATA_DIR,transform=transform)
print(f"\nTotal images: {len(dataset)}"); print(f"Classes: {dataset.classes}")
if dataset.classes != CLASS_NAMES: raise RuntimeError(f"Class order mismatch. Expected {CLASS_NAMES}, found {dataset.classes}.")
if len(dataset)!=28560: print(f"WARNING: Expected 28560 images, found {len(dataset)}.")
if not os.path.isfile(FIXED_FOLDS_FILE): raise FileNotFoundError(f"FIXED_5_FOLD_SPLITS.npz not found:\n{FIXED_FOLDS_FILE}")

folds=[]
with np.load(FIXED_FOLDS_FILE) as d:
    required=[x for k in range(1,6) for x in (f"fold_{k}_train",f"fold_{k}_validation")]
    missing=[x for x in required if x not in d.files]
    if missing: raise KeyError(f"Missing fixed-fold keys: {missing}\nAvailable: {d.files}")
    all_idx=np.arange(len(dataset))
    for k in range(1,6):
        tr=d[f"fold_{k}_train"].astype(np.int64); va=d[f"fold_{k}_validation"].astype(np.int64)
        if len(tr)!=22848 or len(va)!=5712: raise RuntimeError(f"Fold {k} unexpected sizes: {len(tr)}, {len(va)}")
        if np.intersect1d(tr,va).size: raise RuntimeError(f"Fold {k} has train/validation overlap.")
        if not np.all(np.isin(tr,all_idx)) or not np.all(np.isin(va,all_idx)): raise RuntimeError(f"Fold {k} has invalid indices.")
        folds.append((tr,va))
print("\nExisting fixed folds loaded successfully.")
for k,(tr,va) in enumerate(folds,1): print(f"Fold {k}: train={len(tr)}, validation={len(va)}")

def create_model():
    m=efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if PRETRAINED else None)
    m.classifier[1]=nn.Linear(m.classifier[1].in_features,NUM_CLASSES)
    return m

def atomic_save(obj,path):
    tmp=path+".tmp"; torch.save(obj,tmp); os.replace(tmp,path)

def atomic_json(obj,path):
    tmp=path+".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(obj,f,indent=4)
    os.replace(tmp,path)

def save_combined_confusion_matrix(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred,labels=np.arange(NUM_CLASSES))
    table=pd.DataFrame(cm,index=CLASS_NAMES,columns=CLASS_NAMES); table.index.name="Actual"; table.columns.name="Predicted"
    table.to_csv(os.path.join(SAVE_ROOT,"EFFICIENTNET_B3_COMBINED_5FOLD_CONFUSION_MATRIX.csv"))
    fig,ax=plt.subplots(figsize=(8,7)); image=ax.imshow(cm,interpolation="nearest",cmap="Blues"); fig.colorbar(image,ax=ax,fraction=0.046,pad=0.04)
    ax.set(title="EfficientNet-B3 - Combined 5-Fold Confusion Matrix",xlabel="Predicted label",ylabel="Actual label",xticks=np.arange(NUM_CLASSES),yticks=np.arange(NUM_CLASSES),xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES)
    plt.setp(ax.get_xticklabels(),rotation=30,ha="right"); threshold=cm.max()/2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES): ax.text(j,i,f"{cm[i,j]:d}",ha="center",va="center",color="white" if cm[i,j]>threshold else "black")
    fig.tight_layout(); fig.savefig(os.path.join(SAVE_ROOT,"EFFICIENTNET_B3_COMBINED_5FOLD_CONFUSION_MATRIX.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

def rng_state():
    s={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state().tolist()}
    if torch.cuda.is_available(): s["cuda"]=[x.tolist() for x in torch.cuda.get_rng_state_all()]
    return s

def restore_rng(s):
    if not s:return
    try: random.setstate(tuple(s["python"]))
    except: pass
    try:
        n=s["numpy"]; n=(n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]); np.random.set_state(n)
    except: pass
    try: torch.set_rng_state(torch.tensor(s["torch"],dtype=torch.uint8))
    except: pass
    try:
        if torch.cuda.is_available() and "cuda" in s: torch.cuda.set_rng_state_all([torch.tensor(x,dtype=torch.uint8) for x in s["cuda"]])
    except: pass

def metrics(y,p,probs):
    r={"accuracy":accuracy_score(y,p)}; r["micro_accuracy"]=r["accuracy"]; r["macro_class_accuracy"]=balanced_accuracy_score(y,p)
    for avg,prefix in [("macro","macro"),("micro","micro")]:
        r[f"{prefix}_precision"]=precision_score(y,p,average=avg,zero_division=0); r[f"{prefix}_recall"]=recall_score(y,p,average=avg,zero_division=0); r[f"{prefix}_f1"]=f1_score(y,p,average=avg,zero_division=0)
    yb=label_binarize(y,classes=np.arange(NUM_CLASSES))
    for avg in ["macro","micro"]:
        try:r[f"{avg}_auc"]=roc_auc_score(yb,probs,average=avg,multi_class="ovr")
        except:r[f"{avg}_auc"]=np.nan
    return r

@torch.no_grad()
def evaluate(model,loader):
    model.eval(); crit=nn.CrossEntropyLoss(); yt=[]; yp=[]; pr=[]; loss_sum=0; n=0
    for x,y in loader:
        x=x.to(DEVICE,non_blocking=True); y=y.to(DEVICE,non_blocking=True)
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=USE_AMP): out=model(x); loss=crit(out,y)
        prob=torch.softmax(out,1); pred=prob.argmax(1); bs=y.size(0); loss_sum+=loss.item()*bs; n+=bs
        yt.extend(y.cpu().numpy()); yp.extend(pred.cpu().numpy()); pr.append(prob.cpu().numpy())
    yt=np.asarray(yt); yp=np.asarray(yp); pr=np.concatenate(pr); r=metrics(yt,yp,pr); r["loss"]=loss_sum/n; return r,yt,yp,pr

def train_fold(fold,tr_idx,va_idx):
    fd=os.path.join(SAVE_ROOT,f"Fold_{fold}"); cd=os.path.join(fd,"checkpoints"); os.makedirs(cd,exist_ok=True)
    complete=os.path.join(fd,"FOLD_COMPLETE.json"); best=os.path.join(fd,"BEST_MODEL.pth"); latest=os.path.join(cd,"LATEST_CHECKPOINT.pth"); history_path=os.path.join(fd,"training_history.csv")
    if os.path.isfile(complete):
        with open(complete,encoding="utf-8") as f:return json.load(f)["final_metrics"]
    tl=DataLoader(Subset(dataset,tr_idx.tolist()),batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=DEVICE.type=="cuda")
    vl=DataLoader(Subset(dataset,va_idx.tolist()),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=DEVICE.type=="cuda")
    model=create_model().to(DEVICE); crit=nn.CrossEntropyLoss(); opt=optim.Adam(model.parameters(),lr=LEARNING_RATE,foreach=False); scaler=torch.amp.GradScaler("cuda",enabled=USE_AMP)
    start=1; best_acc=-np.inf; best_epoch=0; patience=0; history=[]
    if os.path.isfile(latest):
        print(f"\nLatest checkpoint found for Fold {fold}. Resuming...")
        c=torch.load(latest,map_location="cpu",weights_only=False); model.load_state_dict(c["model_state_dict"]); start=c["epoch"]+1; best_acc=c["best_val_accuracy"]; best_epoch=c["best_epoch"]; patience=c["patience_counter"]; history=c.get("history",[]); restore_rng(c.get("rng_state")); del c
    for epoch in range(start,NUM_EPOCHS+1):
        model.train(); ls=correct=total=0
        for x,y in tqdm(tl,desc=f"Fold {fold} | Epoch {epoch}/{NUM_EPOCHS}",leave=False):
            x=x.to(DEVICE,non_blocking=True); y=y.to(DEVICE,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=USE_AMP): out=model(x); loss=crit(out,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); bs=y.size(0); ls+=loss.item()*bs; correct+=(out.argmax(1)==y).sum().item(); total+=bs
        tr_loss=ls/total; tr_acc=correct/total; vm,_,_,_=evaluate(model,vl); va=float(vm["accuracy"]); history.append({"epoch":epoch,"train_loss":tr_loss,"train_accuracy":tr_acc,"val_loss":vm["loss"],"val_accuracy":va,"val_macro_precision":vm["macro_precision"],"val_macro_recall":vm["macro_recall"],"val_macro_f1":vm["macro_f1"]}); pd.DataFrame(history).to_csv(history_path,index=False)
        if va>best_acc+MIN_DELTA:
            best_acc=va; best_epoch=epoch; patience=0; atomic_save({k:v.detach().cpu() for k,v in model.state_dict().items()},best)
        else: patience+=1
        c={"epoch":epoch,"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"best_val_accuracy":float(best_acc),"best_epoch":best_epoch,"patience_counter":patience,"history":history,"rng_state":rng_state(),"optimizer_state_saved":False}
        # Keep only one replaceable recovery checkpoint. Saving every epoch permanently
        # can consume tens of GB across five folds and was the main storage-crash risk.
        atomic_save(c,latest); del c
        print(f"\nFold {fold} | Epoch {epoch:02d} | Train Acc: {tr_acc:.4f} | Val Acc: {va:.4f} | Best: {best_acc:.4f} | Patience: {patience}/{EARLY_STOPPING_PATIENCE}")
        if patience>=EARLY_STOPPING_PATIENCE: print(f"\nEarly stopping triggered at epoch {epoch}."); break
    model.load_state_dict(torch.load(best,map_location="cpu",weights_only=True)); model.to(DEVICE); fm,y,p,probs=evaluate(model,vl)
    np.savez_compressed(os.path.join(fd,"predictions.npz"),y_true=y,y_pred=p,y_probs=probs)
    cm=confusion_matrix(y,p,labels=np.arange(NUM_CLASSES)); np.savetxt(os.path.join(fd,"confusion_matrix.csv"),cm,delimiter=",",fmt="%d")
    plt.figure(figsize=(7,6)); plt.imshow(cm); plt.title(f"EfficientNet-B3 - Fold {fold}"); plt.xlabel("Predicted"); plt.ylabel("True"); plt.xticks(range(4),CLASS_NAMES,rotation=45,ha="right"); plt.yticks(range(4),CLASS_NAMES)
    for i in range(4):
        for j in range(4): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.savefig(os.path.join(fd,"confusion_matrix.png"),dpi=300); plt.close()
    pd.DataFrame(classification_report(y,p,target_names=CLASS_NAMES,output_dict=True,zero_division=0)).transpose().to_csv(os.path.join(fd,"classification_report.csv"))
    result={"fold":fold,**{k:fm[k] for k in ["accuracy","micro_accuracy","macro_class_accuracy","macro_precision","micro_precision","macro_recall","micro_recall","macro_f1","micro_f1","macro_auc","micro_auc","loss"]},"best_epoch":best_epoch,"best_val_accuracy":best_acc,"validation_samples":len(va_idx)}
    pd.DataFrame([result]).to_csv(os.path.join(fd,"fold_metrics.csv"),index=False); atomic_json({"fold":fold,"status":"complete","final_metrics":result},complete)
    # Predictions are now permanent, so large recovery weights are no longer needed.
    for temporary_weight in (best,latest):
        if os.path.isfile(temporary_weight): os.remove(temporary_weight)
    if os.path.isdir(cd) and not os.listdir(cd): os.rmdir(cd)
    print(f"\nFOLD {fold} COMPLETE")
    del model,opt,scaler,tl,vl
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result

def main():
    start=time.time(); results=[]
    for fold,(tr,va) in enumerate(folds,1):
        print("\n"+"="*80); print(f"EFFICIENTNET-B3 - FOLD {fold}/{N_SPLITS}"); print("="*80); results.append(train_fold(fold,tr,va)); pd.DataFrame(results).to_csv(os.path.join(SAVE_ROOT,"EFFICIENTNET_B3_5FOLD_RESULTS_PROGRESS.csv"),index=False)
    df=pd.DataFrame(results); df.to_csv(os.path.join(SAVE_ROOT,"EFFICIENTNET_B3_5FOLD_RESULTS.csv"),index=False)
    prediction_parts=[]
    for fold in range(1,N_SPLITS+1):
        with np.load(os.path.join(SAVE_ROOT,f"Fold_{fold}","predictions.npz")) as saved:
            part=pd.DataFrame({"fold":fold,"true_label":saved["y_true"],"predicted_label":saved["y_pred"]})
            for class_index,class_name in enumerate(CLASS_NAMES): part[f"prob_{class_name}"]=saved["y_probs"][:,class_index]
            prediction_parts.append(part)
    combined_predictions=pd.concat(prediction_parts,ignore_index=True); combined_predictions.to_csv(os.path.join(SAVE_ROOT,"EFFICIENTNET_B3_ALL_5FOLD_OOF_PREDICTIONS.csv"),index=False)
    save_combined_confusion_matrix(combined_predictions["true_label"].to_numpy(),combined_predictions["predicted_label"].to_numpy())
    cols=["accuracy","micro_accuracy","macro_class_accuracy","macro_precision","micro_precision","macro_recall","micro_recall","macro_f1","micro_f1","macro_auc","micro_auc","loss","best_epoch"]
    summary_rows=[]
    for c in cols:
        values=pd.to_numeric(df[c]); mean=values.mean(); sd=values.std(ddof=1); se=sd/np.sqrt(N_SPLITS)
        summary_rows.append({"metric":c,"mean":mean,"std":sd,"standard_error":se,"ci95_lower":mean-2.7764451051977987*se,"ci95_upper":mean+2.7764451051977987*se})
    summary=pd.DataFrame(summary_rows); summary.to_csv(os.path.join(SAVE_ROOT,"EFFICIENTNET_B3_5FOLD_SUMMARY.csv"),index=False)
    atomic_json({"experiment":"EfficientNet-B3 5-Fold Stratified CV","architecture":"EfficientNet-B3 ImageNet-pretrained","dataset":DATA_DIR,"fixed_fold_file":FIXED_FOLDS_FILE,"total_images":len(dataset),"classes":CLASS_NAMES,"number_of_folds":N_SPLITS,"seed":SEED,"epochs":NUM_EPOCHS,"batch_size":BATCH_SIZE,"learning_rate":LEARNING_RATE,"early_stopping_patience":EARLY_STOPPING_PATIENCE,"num_workers":NUM_WORKERS,"pretrained":PRETRAINED,"device":str(DEVICE),"optimizer":"Adam","optimizer_state_saved":False,"runtime_minutes":(time.time()-start)/60},os.path.join(SAVE_ROOT,"FINAL_EXPERIMENT_RECORD.json"))
    print("\n\n"+"="*80); print("EFFICIENTNET-B3 - FINAL 5-FOLD RESULTS"); print("="*80); print("\nFOLD-WISE RESULTS:\n"); print(df.to_string(index=False)); print("\n\nMEAN ± SD:\n")
    for _,r in summary.iterrows(): print(f"{r['metric']:25s}: {r['mean']:.6f} ± {r['std']:.6f}")
    print(f"\nAll results saved in:\n{SAVE_ROOT}\n\nFixed folds reused from:\n{FIXED_FOLDS_FILE}")

if __name__=="__main__":
    import multiprocessing; multiprocessing.freeze_support(); main()
