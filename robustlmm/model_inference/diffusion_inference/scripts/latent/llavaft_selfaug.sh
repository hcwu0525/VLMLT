source ~/.bashrc
source ~/anaconda3/bin/activate smoe

# environment variables
export OMP_NUM_THREADS=8
AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address

accelerate_config_file=/mnt/petrelfs/songmingyang/.config/accelerate/4gpus.yaml
code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/diffusion_inference
cd $code_base

# export SLURM_JOB_ID=3304670

gpus=4
cpus=32
quotatype="auto"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="diffusion" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch --config_file=${accelerate_config_file}  ./diffusion_inference.py \
--input_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/latent/llavaft/latent_reps/latent_supply/llavaft_selfsupply.jsonl \
--img_dir_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs \
--output_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/latent_aug/llavaft_selfaug \
--batch_size 64 \
--num_workers 4 \
--candidate_img_num 2 \
--function latent_llavaft_self 

