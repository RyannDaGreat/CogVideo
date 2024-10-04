cd /root/CleanCode/Github/CogVideo

DATASET_PATH = "datasets/Disney-VideoGeneration-Dataset"
MODEL_PATH = "models_huggingface/CogVideoX-5b"
OUTPUT_PATH = "outputs_train"
CACHE_DIR = "cache_train"

accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \  # Use accelerate to launch multi-GPU training with the config file accelerate_config_machine_single.yaml
  train_cogvideox_lora.py \  # Training script train_cogvideox_lora.py for LoRA fine-tuning on CogVideoX model
  --cache_dir $CACHE_PATH \  # Cache directory for model files, specified by $CACHE_PATH
  --instance_data_root $DATASET_PATH \  # Dataset path specified by $DATASET_PATH
  --pretrained_model_name_or_path $MODEL_PATH \  # Path to the pretrained model, specified by $MODEL_PATH
  --output_dir $OUTPUT_PATH \  # Specify the output directory for the model, defined by $OUTPUT_PATH
  --validation_prompt "" \  # Prompt used for generating validation videos during training
  --gradient_checkpointing \  # Enable gradient checkpointing to reduce memory usage
  --enable_tiling \  # Enable tiling technique to process videos in chunks, saving memory
  --enable_slicing \  # Enable slicing to further optimize memory by slicing inputs
  --caption_column prompts.txt \  # Specify the file prompts.txt for video descriptions used in training
  --video_column videos.txt \  # Specify the file videos.txt for video paths used in training
  --validation_prompt_separator ::: \  # Set ::: as the separator for validation prompts
  --num_validation_videos 1 \  # Generate 1 validation video per validation round
  --validation_epochs 100 \  # Perform validation every 100 training epochs
  --seed 42 \  # Set random seed to 42 for reproducibility
  --rank 128 \  # Set the rank for LoRA parameters to 128
  --lora_alpha 64 \  # Set the alpha parameter for LoRA to 64, adjusting LoRA learning rate
  --mixed_precision bf16 \  # Use bf16 mixed precision for training to save memory
  --height 480 \  # Set video height to 480 pixels
  --width 720 \  # Set video width to 720 pixels
  --fps 8 \  # Set video frame rate to 8 frames per second
  --max_num_frames 49 \  # Set the maximum number of frames per video to 49
  --skip_frames_start 0 \  # Skip 0 frames at the start of the video
  --skip_frames_end 0 \  # Skip 0 frames at the end of the video
  --train_batch_size 4 \  # Set training batch size to 4
  --num_train_epochs 30 \  # Total number of training epochs set to 30
  --checkpointing_steps 1000 \  # Save model checkpoint every 1000 steps
  --gradient_accumulation_steps 1 \  # Accumulate gradients for 1 step, updating after each batch
  --learning_rate 1e-3 \  # Set learning rate to 0.001
  --lr_scheduler cosine_with_restarts \  # Use cosine learning rate scheduler with restarts
  --lr_warmup_steps 200 \  # Warm up the learning rate for the first 200 steps
  --lr_num_cycles 1 \  # Set the number of learning rate cycles to 1
  --optimizer AdamW \  # Use the AdamW optimizer
  --adam_beta1 0.9 \  # Set Adam optimizer beta1 parameter to 0.9
  --adam_beta2 0.95 \  # Set Adam optimizer beta2 parameter to 0.95
  --max_grad_norm 1.0 \  # Set maximum gradient clipping value to 1.0
  --allow_tf32 \  # Enable TF32 to speed up training
  --report_to wandb  # Use Weights and Biases (wandb) for logging and monitoring the training\