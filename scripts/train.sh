export CUDA_VISIBLE_DEVICES=0
PROJECT=Istree
MODEL=transformer
total_training_sample=100000
training_samples=10000
n_digits=30
k=3
CoT=True
lr=6e-5
num_layers=4
num_heads=3
python src/train.py \
    --world_size 1 \
    --total_training_samples ${training_samples} \
    --model_type transformer \
    --model_config_path config/gpt2_tiny_wpetrain.py \
    --dataset_dir data/Nonintersect_Binary/binary_${training_samples}_${n_digits}_${k}_${CoT}_False_False \
    --dataset_type BinaryDataset \
    --output_dir model/ \
    --batch_size 512 \
    --lr ${lr} \
    --weight_decay 0 \
    --log_interval 2048 \
    --save_interval 2048 \
    --eval_interval 2048 \
    --report_to_wandb \
   --num_hidden_layers ${num_layers} \
   --num_attention_heads ${num_heads} \