#!/bin/bash

# 配置项
INPUT_FILE="jgex_ag_231.txt"
OUTPUT_DIR="results"
SUMMARY_LOG="$OUTPUT_DIR/summary_report.txt"
MODEL_DIR="./mini_ag_weights"
BEAM_SIZE=4
SEARCH_DEPTH=4
# 设置并发线程数（请根据你的显存/内存大小谨慎调整！）
NUM_THREADS=4 

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 找不到输入文件 $INPUT_FILE"
    exit 1
fi

# 创建输出目录并清空旧的状态文件，防止干扰
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*.status

echo "开始执行 Mini-AlphaGeometry 多线程批量测试 (并发数: $NUM_THREADS)..." | tee "$SUMMARY_LOG"
echo "==================================================" | tee -a "$SUMMARY_LOG"

# 定义单道题目的处理函数
run_single_test() {
    prob_name="$1"
    INPUT_FILE="$2"
    OUTPUT_DIR="$3"
    
    # 获取无后缀的文件名
    BASE_NAME=$(basename "$prob_name" .gex)
    LOG_FILE="$OUTPUT_DIR/${BASE_NAME}.log"
    SOL_FILE="$OUTPUT_DIR/${BASE_NAME}_solution.txt"
    STATUS_FILE="$OUTPUT_DIR/${BASE_NAME}.status"
    
    echo "[启动] 正在测试: $prob_name"
    
    # 调用你的 Python 脚本
    uv run python littlegeometry.py \
        --problems_file="${INPUT_FILE}" \
        --problem_name="${prob_name}" \
        --model_dir="${MODEL_DIR}" \
        --out_file="${SOL_FILE}" \
        --beam_size=${BEAM_SIZE} \
        --search_depth=${SEARCH_DEPTH} \
        > "$LOG_FILE" 2>&1
        
    # 分析日志结果，写入状态文件
    if grep -q "求解完成" "$LOG_FILE"; then
        echo "DDAR_ONLY" > "$STATUS_FILE"
        echo "[完成] $prob_name -> 纯符号引擎解出"
    elif grep -q "解题成功" "$LOG_FILE"; then
        echo "WITH_AUX" > "$STATUS_FILE"
        echo "[完成] $prob_name -> 使用辅助点解出"
    else
        echo "FAILED" > "$STATUS_FILE"
        echo "[失败] $prob_name -> 未能找出证明"
    fi
}

# 导出函数，使其可以在 xargs 的子 shell 中被调用
export -f run_single_test

# 提取题目并使用 xargs 并行执行
# -P 指定并发数，-n 1 每次传递 1 个参数，-I {} 将参数替换到命令中
grep "\.gex$" "$INPUT_FILE" | xargs -n 1 -P "$NUM_THREADS" -I {} bash -c 'run_single_test "{}" "'"$INPUT_FILE"'" "'"$OUTPUT_DIR"'"'

echo "==================================================" | tee -a "$SUMMARY_LOG"
echo "所有线程执行完毕，正在汇总结果..." | tee -a "$SUMMARY_LOG"

# 统计各个状态文件中的结果
TOTAL_PROBLEMS=$(ls "$OUTPUT_DIR"/*.status 2>/dev/null | wc -l)
SOLVED_DDAR_ONLY=$(grep -h "DDAR_ONLY" "$OUTPUT_DIR"/*.status 2>/dev/null | wc -l)
SOLVED_WITH_AUX=$(grep -h "WITH_AUX" "$OUTPUT_DIR"/*.status 2>/dev/null | wc -l)
FAILED=$(grep -h "FAILED" "$OUTPUT_DIR"/*.status 2>/dev/null | wc -l)

TOTAL_SOLVED=$((SOLVED_DDAR_ONLY + SOLVED_WITH_AUX))

# 打印最终统计信息
echo "### 测试统计报告 ###" | tee -a "$SUMMARY_LOG"
echo "总测试题目数:      $TOTAL_PROBLEMS" | tee -a "$SUMMARY_LOG"
echo "总计成功解答:      $TOTAL_SOLVED" | tee -a "$SUMMARY_LOG"
echo "--------------------------------------------------" | tee -a "$SUMMARY_LOG"
echo "纯符号推导解答数:  $SOLVED_DDAR_ONLY" | tee -a "$SUMMARY_LOG"
echo "使用辅助点解答数:  $SOLVED_WITH_AUX" | tee -a "$SUMMARY_LOG"
echo "未能解答数:        $FAILED" | tee -a "$SUMMARY_LOG"
echo "==================================================" | tee -a "$SUMMARY_LOG"

