#!/bin/bash
# run_predict.sh
# 使用方法: bash run_predict.sh <fasta_file> <target_ion> <save_dir>
# 示例 bash run_predict.sh ./test.fasta zn .
# 结果文件： os.path.join(save_dir, predict.csv)

cd /home/ysq/需备份文件/T5模型参数 || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_2.14

FASTA_FILE=$1
TARGET_ION=$2
SAVE_DIR=$3

if [ -f "./temp.tfrecord" ]; then
    echo "删除旧的 temp.tfrecord 文件..."
    rm ./temp.tfrecord
fi

python pro_process.py -f "$FASTA_FILE" -t "$TARGET_ION"

python predict.py -f ./temp.tfrecord -t "$TARGET_ION" -s "$SAVE_DIR"
