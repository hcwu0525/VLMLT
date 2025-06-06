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

# export SLURM_JOB_ID=3442208
aug_output_path=/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/share4v_aug/aug_ft_200k_tocw
mkdir -p $aug_output_path

gpus=8
cpus=64
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="diffusion" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch --config_file=${accelerate_config_file}  ./diffusion_inference.py \
--input_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/origin_data_full_info/share_ft_dinoobj.json \
--img_dir_path /mnt/petrelfs/share_data/quxiaoye/sharegpt4v/data \
--output_path $aug_output_path \
--batch_size 64 \
--num_workers 4 \
--origin_alpha 1 \
--augment_alpha 1 \
--pass_num 0 \
--candidate_img_num 2 \
--function share4v_ft 


