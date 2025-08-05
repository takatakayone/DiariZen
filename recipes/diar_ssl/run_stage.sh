#!/bin/bash
# DiariZen diar_ssl run script (fixed for single-GPU JupyterHub node)

set -e
ulimit -n 2048

export MKL_INTERFACE_LAYER=""                                
source "$(conda info --base)/etc/profile.d/conda.sh"        
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
recipe_root="$SCRIPT_DIR"
exp_root="$recipe_root/exp"
conf_dir="$recipe_root/conf"
embedding_model=$recipe_root/../pretrained_models/pytorch_model.bin
dscore_dir=/home/jovyan/DiariZen/dscore
conda_env=/home/jovyan/my_env
stage=1
config_suffix=""

[ $# -ge 1 ] && stage=$1  
[ $# -ge 2 ] && config_suffix=$2

# 設定ファイル名の決定
if [ -n "$config_suffix" ]; then
    train_conf=$conf_dir/wavlm_updated_conformer-${config_suffix}.toml
else
    # train_conf=$conf_dir/wavlm_updated_conformer.toml
    train_conf=$conf_dir/wavlm_samba.toml
fi

use_dual_opt=true

conf_name=$(basename "${train_conf%.*}")

dtype=test
data_dir=$recipe_root/data/AMI_AliMeeting_AISHELL4
seg_duration=8

clustering_method=AgglomerativeClustering
ahc_threshold=0.70
min_cluster_size=30
infer_affix=_constrained_AHC_thres_${ahc_threshold}_mcs_${min_cluster_size}

avg_ckpt_num=5
val_metric=Loss
val_mode=best
collar=0.25

# ============================ stage 1 : train ====================
if [ "$stage" -le 1 ]; then
    if ! $use_dual_opt; then
        echo "stage1: use single-opt for model training..."
        conda activate $conda_env
        CUDA_VISIBLE_DEVICES="0" accelerate launch \
            --num_processes 1 --main_process_port 1134 \
            run_single_opt.py -C "$train_conf" -M validate
    else
        echo "stage1: use dual-opt for model training..."
        conda activate $conda_env
        CUDA_VISIBLE_DEVICES="0" accelerate launch \
            --num_processes 1 --main_process_port 1134 \
            run_dual_opt.py -C "$train_conf" -M train
    fi
fi

# ================== stage 2 : inference & scoring ================
diarization_dir=$exp_root/$conf_name
config_dir=$(ls "$diarization_dir"/*.toml | sort -r | head -n 1)

if [ "$stage" -le 2 ]; then
    echo "stage2: model inference..."
    conda activate $conda_env
    export CUDA_VISIBLE_DEVICES=0

    train_log=$(du -h "$diarization_dir"/*.log | sort -rh | head -n 1 | awk '{print $NF}')
    grep 'Loss/DER' "$train_log" | awk -F ']:' '{print $NF}' > "$diarization_dir/val_metric_summary.lst"

    for dset in AMI AliMeeting AISHELL4; do
        python infer_avg.py -C "$config_dir" \
            -i "${data_dir}/${dtype}/${dset}/wav.scp" \
            -o "${diarization_dir}/infer${infer_affix}/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dtype}/${dset}" \
            --embedding_model "$embedding_model" \
            --avg_ckpt_num "$avg_ckpt_num" \
            --val_metric "$val_metric" \
            --val_mode "$val_mode" \
            --val_metric_summary "$diarization_dir/val_metric_summary.lst" \
            --seg_duration "$seg_duration" \
            --clustering_method "$clustering_method" \
            --ahc_threshold "$ahc_threshold" \
            --min_cluster_size "$min_cluster_size"

        echo "stage3: scoring..."
        SYS_DIR="${diarization_dir}/infer${infer_affix}/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}"
        OUT_DIR="${SYS_DIR}/${dtype}/${dset}"
        python "${dscore_dir}/score.py" \
            -r "${data_dir}/${dtype}/${dset}/rttm" \
            -s "$OUT_DIR"/*.rttm --collar "$collar" \
            > "$OUT_DIR/result_collar${collar}"
    done
fi