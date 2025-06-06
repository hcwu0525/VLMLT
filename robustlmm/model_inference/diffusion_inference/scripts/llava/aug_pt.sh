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

accelerate_config_file=/mnt/petrelfs/songmingyang/.config/accelerate/8gpus_1.yaml
code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/diffusion_inference
cd $code_base

export SLURM_JOB_ID=3304670

gpus=8
cpus=64
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="diffusion" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch --config_file=${accelerate_config_file}  ./diffusion_inference.py \
--input_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reformed_data/origin_data/llava_v1_5_mix665k.json \
--img_dir_path /mnt/petrelfs/share_data/quxiaoye/sharegpt4v/data \
--output_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/share4v_aug/aug_pt_100k \
--batch_size 64 \
--num_workers 4 \
--origin_alpha 0.8 \
--augment_alpha 0.9 \
--pass_num 1 \
--candidate_img_num 2

