def dict_to_name(d=None,**kwargs):
    if d is None:
        d={}
    d.update(kwargs)
    return '_'.join('='.join(map(str,[key,value])) for key,value in d.items())

import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video, load_image
from icecream import ic
import rp.git.CommonSource.noise_warp as nw

device=select_torch_device(prefer_used=True)
dtype=torch.bfloat16

prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
prompt = "An old house by the lake with wooden plank siding and a thatched roof"
image = load_image(image=download_url_to_cache("https://media.sciencephoto.com/f0/22/69/89/f0226989-800px-wm.jpg"))

#prompt='A bowling ball slowly rolling'

if 'pipe' not in vars():
    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b",torch_dtype=torch.bfloat16) ; pipe_name='T2V'
    # pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=dtype) ; pipe_name='I2V'

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

B, F, C, H, W = 1, 13, 16, 60, 90  # The defaults
#F = 12
#H-=1
latents = torch.randn((B, F, C, H, W), device=device, dtype=dtype)

#https://medium.com/@ChatGLM/open-sourcing-cogvideox-a-step-towards-revolutionizing-video-generation-28fa4812699d

num_frames=(F-1)*4+1 #https://miro.medium.com/v2/resize:fit:1400/format:webp/0*zxsAG1xks9pFIsoM
#Possible num_frames: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49

#num_inference_steps = 5 #USER INPUT++
num_inference_steps = 1 #USER INPUT++
#num_inference_steps = 15 #USER INPUT++
#num_inference_steps = 30 #USER INPUT++
#
#FUCK WITH THE LATENTS
for i in range(F):
    latents[0,i]=latents[0,0].roll(10*i,dims=-2)
degrade=0
#degrade=1
latents=nw.mix_new_noise(latents,degrade)

time=millis()

output_name=dict_to_name(gather_vars('time pipe_name num_frames num_inference_steps degrade F H W'))+'.mp4'
output_folder=make_directory('outputs')
output_path=get_unique_copy_path(path_join(output_folder,output_name))

ic(pipe_name, num_frames, num_inference_steps, output_path, degrade, prompt, device)


video = pipe(
    prompt=prompt,
    #image=image,
    num_videos_per_prompt=1,
    height=H*8,
    width=W*8,
    num_inference_steps=num_inference_steps,
    num_frames=num_frames,
    guidance_scale=6,
    generator=torch.Generator(device=device).manual_seed(42),
    latents=latents,
).frames[0]

export_to_video(video, output_path, fps=8)
