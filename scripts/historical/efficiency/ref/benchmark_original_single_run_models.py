"""Count and time original single-run checkpoints; no CV, training, or data changes."""
from pathlib import Path
import ast,csv,gc,hashlib,json,platform,subprocess,time
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision
from torchvision import models,transforms

ROOT=Path(r'C:/user/juli/thesis/original_file/othr folders')
OUT=ROOT/'Reviewer 2 Comment 8 - Single Run Computational Efficiency'
SPECS=[('Proposed',256,'training_outputs_proposed_model_orgnl/training_outputs/hybrid_cnn_bilstm_attention_final.pth'),
 ('VGG16',224,'vgg16_outputs/vgg16_best.pth'),('ResNet50',224,'resnet50_outputs/resnet50_best.pth'),
 ('DenseNet121',224,'densenet_outputs/densenet121_best.pth'),('EfficientNet-B3',300,'efficientnet_b3_outputs/efficientnet_b3_best.pth')]

def snapshot():
 try:return subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu','--format=csv'],text=True).strip()
 except Exception as e:return str(e)

def build(name):
 if name=='Proposed':
  source=Path(__file__).with_name('train_model_withCustomized_cnn.py')
  names={'ResidualBlock','CNNFeatureExtractor','AdditiveAttention','HybridCNNBiLSTMAttention'}
  tree=ast.parse(source.read_text(encoding='utf-8-sig'))
  module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names],type_ignores=[])
  ns={'torch':torch,'nn':nn,'NUM_CLASSES':4};exec(compile(module,str(source),'exec'),ns)
  return ns['HybridCNNBiLSTMAttention']()
 if name=='VGG16':m=models.vgg16(weights=None);m.classifier[6]=nn.Linear(4096,4)
 elif name=='ResNet50':m=models.resnet50(weights=None);m.fc=nn.Linear(2048,4)
 elif name=='DenseNet121':m=models.densenet121(weights=None);m.classifier=nn.Linear(1024,4)
 else:m=models.efficientnet_b3(weights=None);m.classifier[1]=nn.Linear(1536,4)
 return m

def main():
 if not torch.cuda.is_available():raise RuntimeError('GPU required for this protocol')
 OUT.mkdir(exist_ok=True)
 torch.set_num_threads(1);torch.manual_seed(42)
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 paths=[]
 for folder in sorted((ROOT/'preprocessed_testt').iterdir()):
  if folder.is_dir():paths.extend(sorted(folder.glob('*.jpg'))[:8])
 assert len(paths)==32
 metadata=dict(device=torch.cuda.get_device_name(0),pytorch=torch.__version__,torchvision=torchvision.__version__,
  cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),os=platform.platform(),python=platform.python_version(),
  batch_size=1,dtype='float32',tf32=False,cudnn_benchmark=False,cpu_threads=1,warmup_per_model=50,
  timing='Synchronized wall-clock model forward latency; input already resident on GPU; includes dispatch and completion wait.',
  excluded='Disk I/O, image preprocessing, CPU-GPU input transfer, softmax, Grad-CAM/LIME and training.',
  repeats=3,passes_per_repeat=100,inputs=[str(p) for p in paths],
  limitations='Current desktop GPU is shared, not exclusively reserved. Latency is environment-specific; current GPU differs from manuscript training hardware.',
  checkpoints=[],gpu_before=snapshot())
 results=[];raw=[]
 for name,size,relative in SPECS:
  print('Benchmarking',name,flush=True)
  p=ROOT/relative
  saved=torch.load(p,map_location='cpu',weights_only=True)
  state=saved.get('model_state_dict',saved.get('state_dict',saved))
  model=build(name);model.load_state_dict(state,strict=True);del saved,state
  params=sum(p.numel() for p in model.parameters())
  model=model.eval().cuda()
  mean,std=((.5,)*3,(.5,)*3) if name=='Proposed' else ((.485,.456,.406),(.229,.224,.225))
  transform=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize(mean,std)])
  inputs=[]
  for path in paths:
   with Image.open(path) as image:inputs.append(transform(image).unsqueeze(0).cuda())
  latencies=[]
  with torch.inference_mode():
   for i in range(50):prediction=model(inputs[i%len(inputs)])
   logits=prediction[0] if isinstance(prediction,tuple) else prediction
   assert logits.shape==(1,4) and torch.isfinite(logits).all()
   torch.cuda.synchronize()
   for repeat in range(3):
    for i in range(100):
     x=inputs[(repeat*100+i)%len(inputs)]
     torch.cuda.synchronize();start=time.perf_counter_ns()
     prediction=model(x)
     torch.cuda.synchronize();ms=(time.perf_counter_ns()-start)/1e6
     latencies.append(ms);raw.append(dict(model=name,repeat=repeat+1,iteration=i+1,latency_ms=ms))
  arr=np.array(latencies)
  row=dict(model=name,input_shape=f'1x3x{size}x{size}',parameters=params,parameters_millions=params/1e6,
   latency_mean_ms=float(arr.mean()),latency_sd_ms=float(arr.std(ddof=1)),latency_median_ms=float(np.median(arr)),
   latency_p95_ms=float(np.percentile(arr,95)),timed_passes=len(arr))
  results.append(row);print(row,flush=True)
  metadata['checkpoints'].append(dict(model=name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),gpu_after=snapshot()))
  del model,inputs,prediction,logits,x;gc.collect();torch.cuda.empty_cache()
 for filename,rows in [('01_Parameter_Counts_and_Inference_Times.csv',results),('02_All_Individual_Timings.csv',raw)]:
  with (OUT/filename).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 metadata['gpu_after']=snapshot()
 (OUT/'03_Benchmark_Protocol_and_Checkpoints.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
 print('Saved:',OUT,flush=True)

if __name__=='__main__':main()
