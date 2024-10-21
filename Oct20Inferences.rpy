while True:
    # 2024-10-20 00:21:16.198570
    # 2024-10-17 06:39:14.960276
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
    
    #prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
    prompt = "An old house by the lake with wooden plank siding and a thatched roof"
    #image = load_image(image=download_url_to_cache("https://media.sciencephoto.com/f0/22/69/89/f0226989-800px-wm.jpg"))
    
    #prompt = "People riding a bicycles at high speed. Focused, detailed, realistic."
    #image = load_image(image=download_url_to_cache("https://st2.depositphotos.com/2117297/48118/i/450/depositphotos_481187768-stock-photo-exuberant-family-riding-bicycles-forest.jpg"))
    
    #prompt='A bowling ball slowly rolling'
    
    if 'pipe' not in vars():
    
        pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b",torch_dtype=torch.bfloat16) ; pipe_name='T2V'
        print(end="LOADING 5B LORA WEIGHTS...");pipe.load_lora_weights('/root/CleanCode/Github/CogVideo/finetune/cogvideox5b-lora-single-node-delegator-noisewarp-Oct16-RandomDegradation-LargerBatchSize-SmallLearnRate/checkpoint-9600/saved_weights_copy/pytorch_lora_weights.safetensors');print("DONE!")
        
        #pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b",torch_dtype=torch.bfloat16) ; pipe_name='T2V'
        #print(end="LOADING 2B LORA WEIGHTS...");pipe.load_lora_weights('/root/CleanCode/Github/CogVideo/finetune/cogvideox2b-lora-single-node-delegator-noisewarp-Oct16-RandomDegradation-LargerBatchSize-SmallLearnRate/checkpoint-16400/saved_weights_copy/pytorch_lora_weights.safetensors');print("DONE!")
        
        # pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=dtype) ; pipe_name='I2V'
        
    
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    
    #torch.manual_seed(42)
    
    B, F, C, H, W = 1, 13, 16, 60, 90  # The defaults
    #F = 12
    #H-=1
    latents = torch.randn((B, F, C, H, W), device=device, dtype=dtype)
    
    #https://medium.com/@ChatGLM/open-sourcing-cogvideox-a-step-towards-revolutionizing-video-generation-28fa4812699d
    
    num_frames=(F-1)*4+1 #https://miro.medium.com/v2/resize:fit:1400/format:webp/0*zxsAG1xks9pFIsoM
    #Possible num_frames: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49
    
    #num_inference_steps = 5 #USER INPUT++
    # num_inference_steps = 1 #USER INPUT++
    #num_inference_steps = 15 #USER INPUT++
    num_inference_steps = 30 #USER INPUT++
    #num_inference_steps = 30 #USER INPUT++
    num_inference_steps = 50 #USER INPUT++
    #
    #FUCK WITH THE LATENTS
    
    # #ORIGINAL TYPES
    # for i in range(1,F):
    #     #latents[:,i]/=latents[:,i].std(2,keepdim=True) #Normalize them by their channel variances
    #     latents[0,i]=latents[0,i-1].roll(10,dims=-2)
    #     latents[0,i][:10]=torch.randn_like(latents[0,i][:10]) #Don't loop! We got full 3d attention!
    
    #NON-LOOPING ROLL (BTCHW Form) https://tinyurl.com/2ywlk6w4 for demo
    
    if 1:
        for i in range(1,F):
            #LEFT / RIGHT
            ROLL=5
            latents[:,i,:,:,ROLL:]=latents[:,i-1,:,:,:-ROLL]
            latents[:,i,:,:,:ROLL]=torch.randn_like(latents[:,i,:,:,:ROLL])
    
    if 1:
        for i in range(1,F):
            #UP / DOWN
            ROLL=5
            latents[:,i,:,ROLL:,:]=latents[:,i-1,:,:-ROLL,:]
            latents[:,i,:,:ROLL,:]=torch.randn_like(latents[:,i,:,:ROLL,:])
        
    if 1:
        #COMPLETELY FROM SAMPLE: Generate with /root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidSampleGenerator.ipynb
        sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/plus_pug.pkl'
        sample_path = rp.random_element(glob.glob('/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/*.pkl'))
        sample_gif = sample_path+'.gif'
        print(end="LOADING "+sample_path+"...")
        sample=rp.file_to_object(sample_path)
        print("DONE!")
    
        prompt=sample.instance_prompt
        latents=resize_list(sample.instance_noise.to(latents.dtype).to(latents.device), latents.shape[1])[None]
        ic(latents.shape)
        
    
    #DEGREDATION
    #degrade=0
    #degrade=.5
    #degrade=.8
    degrade=.6
    degrade=0
    #degrade=1
    latents=nw.mix_new_noise(latents,degrade)
    
    #END FUCK WITH LATENTS
        
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
    
    #Save the video
    export_path=path_join(make_directory('OUTPUTS/warp_simple2B_notloop/'+get_file_name(sample_path,False)),output_path)
    make_directory(get_path_parent(export_path))
    export_to_video(video, export_path, fps=8)
    
    sample_gif=load_video(sample_gif)
    video=as_numpy_images(video)
    prevideo=horizontally_concatenated_videos(resize_list(video,len(sample_gif)),sample_gif)
    mp4_path=rp.save_video_mp4(prevideo,export_path+'preview.mp4',framerate=16)
    convert_to_gif_via_ffmpeg(mp4_path,framerate=16)