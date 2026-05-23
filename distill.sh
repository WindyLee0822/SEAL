CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --port 8000

CUDA_VISIBLE_DEVICES=1 python distill_main.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir  /nobackup2/windy/verl-agent-master/checkpoints/distill/qwen1.5b_webshop \
  --learning_rate 2e-5 \
  --num_train_epochs 2