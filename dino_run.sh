 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=2452 train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --batch-size 256 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=4 \
  --output-dir="fpo_exp_t95" \
  --exp-name="temp10_ratio100_aug2" \
  --use-fisher-weighting \
  --fisher-ratio 1.0 \
  --fisher-temperature 1.0 \
  --data-dir=/home/user/code/zty/repa_dataset \
  --fisher-aug 2.0 \
  --fisher-time-min 0.95 \
  --fisher-time-max 1.0 \
  --use-time-conditional-fisher \

torchrun --nnodes=1 --nproc_per_node=2 generate.py \
  --model SiT-B/2 \
  --num-fid-samples 50000 \
  --ckpt ./fpo_exp_t95/temp10_ratio100_aug2/checkpoints/0400000.pt \
  --sample-dir ./fpo_exp_t95/temp10_ratio100_aug2/uncfg_samples \
  --path-type=linear \
  --encoder-depth=4 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.0 \

cd /home/user/code/zty/cvpr26/divide_grpo/evaluations
/home/user/.conda/envs/zty_mm/bin/python evaluator.py \
--ref_batch /home/user/code/zty/public_ckpt/fid_stat/VIRTUAL_imagenet256_labeled.npz \
--sample_batch /home/user/code/zty/cvpr26/icml_research/fpo_code/fpo_exp_t95/temp10_ratio100_aug2/uncfg_samples/SiT-B-2-0400000-size-256-vae-ema-cfg-1.0-seed-0-sde.npz
