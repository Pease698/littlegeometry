#!/bin/bash
set -e

PROBLEM_NAME="translated_imo_2004_p1"
PROBLEMS_FILE="imp_ag_30.txt"

echo "====================================="
echo "正在尝试求解几何题: $PROBLEM_NAME"
echo "读取的题目文件为: $PROBLEMS_FILE"
echo "====================================="

# 配置路径和环境变量
DATA=ag_ckpt_vocab
MELIAD_PATH=meliad_lib/meliad
export PYTHONPATH=$PYTHONPATH:$MELIAD_PATH:origin:traindata:newmodel

# 各种参数配置
DDAR_ARGS=(--defs_file=$(pwd)/defs.txt --rules_file=$(pwd)/rules.txt)
SEARCH_ARGS=(--beam_size=2 --search_depth=2)
LM_ARGS=(
  --ckpt_path=$DATA
  --vocab_path=$DATA/geometry.757.model
  --gin_search_paths=$MELIAD_PATH/transformer/configs
  --gin_file=base_htrans.gin
  --gin_file=size/medium_150M.gin
  --gin_file=options/positions_t5.gin
  --gin_file=options/lr_cosine_decay.gin
  --gin_file=options/seq_1024_nocache.gin
  --gin_file=geometry_150M_generate.gin
  --gin_param=DecoderOnlyLanguageModelGenerate.output_token_losses=True
  --gin_param=TransformerTaskConfig.batch_size=2
  --gin_param=TransformerTaskConfig.sequence_length=128
  --gin_param=Trainer.restore_state_variables=False
)

# 直接使用 uv run 在现有环境中执行推理
# 注意：这里使用 $(pwd)/$PROBLEMS_FILE 确保它能在当前目录下找到你的文件
uv run python -m alphageometry \
  --alsologtostderr \
  --problems_file=$(pwd)/$PROBLEMS_FILE \
  --problem_name=$PROBLEM_NAME \
  --mode=alphageometry \
  "${DDAR_ARGS[@]}" "${SEARCH_ARGS[@]}" "${LM_ARGS[@]}"

