#!/bin/bash

PROBLEMS_FILE="imo_ag_30.txt"
PROBLEM_NAME="translated_imo_2004_p1"
MODEL_DIR="./mini_ag_weights"
OUT_FILE="solution_${PROBLEM_NAME}.txt"
BEAM_SIZE=4
SEARCH_DEPTH=4

echo "==========================================================="
echo "正在启动神经符号求解引擎..."
echo "题库文件: ${PROBLEMS_FILE}"
echo "目标题目: ${PROBLEM_NAME}"
echo "模型路径: ${MODEL_DIR}"
echo "==========================================================="

uv run python littlegeometry.py \
    --problems_file="${PROBLEMS_FILE}" \
    --problem_name="${PROBLEM_NAME}" \
    --model_dir="${MODEL_DIR}" \
    --out_file="${OUT_FILE}" \
    --beam_size=${BEAM_SIZE} \
    --search_depth=${SEARCH_DEPTH}

# 4. 运行结束后的状态提示
if [ $? -eq 0 ]; then
    echo "运行结束"
else
    echo "运行过程中出现错误，请检查报错日志"
fi

