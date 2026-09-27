"""GPU smoke with a public-domain still and synthetic tone, not a lip-sync quality demo."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import hashlib
import subprocess
import wave
import cv2
import gc

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--comfy-root',required=True,type=Path)
parser.add_argument('--vae',required=True,type=Path)
parser.add_argument('--steps',type=int,default=1)
args=parser.parse_args()
if args.steps < 1: parser.error('--steps must be positive')
repo=Path(__file__).resolve().parent
sys.path.insert(0,str(args.comfy_root))
spec=importlib.util.spec_from_file_location('gap_candidate',repo/'__init__.py',submodule_search_locations=[str(repo)])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
spec.loader.exec_module(m)
print('NODE_IMPORT_OK',list(m.NODE_CLASS_MAPPINGS),flush=True)
import torch
import folder_paths
import comfy.sd
import comfy.utils
from skimage import data
folder_paths.set_temp_directory(str(repo/'debug_output'))
(repo/'debug_output').mkdir(exist_ok=True)
folder_paths.add_model_folder_path('latentsync',str(args.comfy_root/'models/latentsync'))
vae=comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(args.vae),safe_load=True))
models=m.NODE_CLASS_MAPPINGS['GapLipSyncModelLoader']().load_models('latentsync_unet.pt','tiny','fp16',vae)[0]
print('MODEL_LOAD_OK',flush=True)
# NASA image via skimage data: public domain. A static 16-frame tensor is a pipeline fixture.
frames=torch.from_numpy(data.astronaut().copy()).float().div(255).unsqueeze(0).repeat(16,1,1,1)
sr=16000
samples=torch.arange(int(sr*16/25))/sr
audio={'waveform':(0.05*torch.sin(2*torch.pi*220*samples)).reshape(1,1,-1),'sample_rate':sr}
torch.cuda.reset_peak_memory_stats()
started=time.perf_counter()
result=m.NODE_CLASS_MAPPINGS['GapLipSyncSampler']().sample(models,frames,audio,1247,args.steps,1.5,25,'512',0,0)
assert tuple(result[0].shape) == (16,512,512,3), result[0].shape
assert torch.isfinite(result[0]).all()
print(json.dumps({'status':'GPU_TENSOR_SMOKE_COMPLETED','output_shape':list(result[0].shape),'seconds':time.perf_counter()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'steps':args.steps,'speech_quality_verified':False}),flush=True)

video_path=repo/'debug_output'/'smoke-silent.mp4'
writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),25,(512,512))
if not writer.isOpened(): raise RuntimeError('Video encoder did not open')
for frame in result[0]:
    pixels=(frame.clamp(0,1).cpu().numpy()*255).astype('uint8')
    writer.write(cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
writer.release()
audio_path=repo/'debug_output'/'smoke-tone.wav'
with wave.open(str(audio_path),'wb') as stream:
    stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(sr)
    stream.writeframes((audio['waveform'].flatten().numpy()*32767).astype('<i2').tobytes())
output=repo/'debug_output'/'smoke.mp4'
subprocess.run(['ffmpeg','-y','-v','error','-i',str(video_path),'-i',str(audio_path),'-c:v','copy','-c:a','aac','-shortest',str(output)],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,nb_frames,width,height','-of','json',str(output)]))
print(json.dumps({'export':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'probe':probe,'fixture':'NASA public-domain astronaut + synthesized 220Hz tone','quality_claim':False}),flush=True)
del result,models,vae
gc.collect()
