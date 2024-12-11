CUDA_VISIBLE_DEVICES="0,1,2,3," accelerate launch train_ccsr_stage1.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-1-base" \
--controlnet_model_name_or_path='preset/models/pretrained_controlnet' \
--enable_xformers_memory_efficient_attention \
--output_dir="./experiments/ccsrv2_stage1" \
--mixed_precision="fp16" \
--resolution=512 \
--learning_rate=5e-5 \
--train_batch_size=4 \
--gradient_accumulation_steps=6 \
--dataloader_num_workers=0 \
--checkpointing_steps=500 \
--t_max=0.6667 \
--max_train_steps=20000 \
--dataset_root_folders 'preset/gt_path.txt' 