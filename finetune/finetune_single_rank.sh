export OUTPUT_SUFFIX="-lora-noisewarp-Oct23-LORA2048-RandDegrad-BlendNoiseNorm"

# export MODEL_PATH="THUDM/CogVideoX-2b"
# export OUTPUT_PATH="cogvideox2b$OUTPUT_SUFFIX"

export MODEL_PATH="THUDM/CogVideoX-5b" 
export OUTPUT_PATH="cogvideox5b$OUTPUT_SUFFIX"

# export DATASET_PATH="/root/CleanCode/Github/CogVideo/finetune/datasets/Disney-VideoGeneration-Dataset"
export DATASET_PATH="/root/CleanCode/Github/CogVideo/finetune/datasets/Single-Sample-Disney-VideoGeneration-Dataset"

#DEFAULT VALUES:
export RANK=128
export LORA_ALPHA=64

#(LORA_ALPHA / RANK) = the lora's strength on a scale from 0 to 1
#https://datascience.stackexchange.com/questions/123229/understanding-alpha-parameter-tuning-in-lora-paper
export RANK=2048
export LORA_ALPHA=2048

#Idk what these do
export CACHE_PATH="~/.cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \
  train_cogvideox_lora.py \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 30000 \
  --checkpointing_steps 200 \
  --dataloader_num_workers 0 \
  --rank $RANK \
  --lora_alpha $LORA_ALPHA \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --validation_prompt "DISNEY A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions:::A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 100 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb
