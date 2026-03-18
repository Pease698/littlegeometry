#!/bin/bash

# 配置项
INPUT_FILE="jgex_ag_231.txt"
OUTPUT_DIR="results"
SUMMARY_LOG="$OUTPUT_DIR/summary_report.txt"
MODEL_DIR="./mini_ag_weights"
BEAM_SIZE=4
SEARCH_DEPTH=6

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 找不到输入文件 $INPUT_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 初始化统计变量
TOTAL_PROBLEMS=0
SOLVED_DDAR_ONLY=0
SOLVED_WITH_AUX=0
FAILED=0

echo "开始执行 Mini-AlphaGeometry 批量测试..." | tee "$SUMMARY_LOG"
echo "==================================================" | tee -a "$SUMMARY_LOG"

# 提取题目名称。根据 jgex_ag_231.txt 的格式，题目名通常是带有 .gex 的文件路径行
# 我们使用进程替换 < <(...) 确保循环内部的变量修改在循环外部生效
while read -r prob_name; do
    # 跳过空行
    [ -z "$prob_name" ] && continue
    
    TOTAL_PROBLEMS=$((TOTAL_PROBLEMS + 1))
    
    # 格式化输出日志文件名 (取路径的最后一部分作为文件名)
    BASE_NAME=$(basename "$prob_name" .gex)
    LOG_FILE="$OUTPUT_DIR/${BASE_NAME}.log"
    SOL_FILE="$OUTPUT_DIR/${BASE_NAME}_solution.txt"
    
    echo "[$TOTAL_PROBLEMS] 正在测试: $prob_name"
    
    # 调用你的 Python 脚本
    uv run python littlegeometry.py \
        --problems_file="${INPUT_FILE}" \
        --problem_name="${prob_name}" \
        --model_dir="${MODEL_DIR}" \
        --out_file="${SOL_FILE}" \
        --beam_size=${BEAM_SIZE} \
        --search_depth=${SEARCH_DEPTH} \
        > "$LOG_FILE" 2>&1
        
    # 分析日志结果，匹配 littlegeometry.py 中的 logging 文本
    if grep -q "求解完成" "$LOG_FILE"; then
        echo "  -> [成功] 纯符号引擎解出" | tee -a "$SUMMARY_LOG"
        SOLVED_DDAR_ONLY=$((SOLVED_DDAR_ONLY + 1))
        
    elif grep -q "解题成功" "$LOG_FILE"; then
        echo "  -> [成功] 使用语言模型辅助点解出" | tee -a "$SUMMARY_LOG"
        SOLVED_WITH_AUX=$((SOLVED_WITH_AUX + 1))
        
    else
        echo "  -> [失败] 未能找出证明" | tee -a "$SUMMARY_LOG"
        FAILED=$((FAILED + 1))
    fi

done < <(grep "\.gex$" "$INPUT_FILE")

# 计算总解答数
TOTAL_SOLVED=$((SOLVED_DDAR_ONLY + SOLVED_WITH_AUX))

# 打印最终统计信息
echo "==================================================" | tee -a "$SUMMARY_LOG"
echo "### 测试统计报告 ###" | tee -a "$SUMMARY_LOG"
echo "总测试题目数:      $TOTAL_PROBLEMS" | tee -a "$SUMMARY_LOG"
echo "总计成功解答:      $TOTAL_SOLVED" | tee -a "$SUMMARY_LOG"
echo "--------------------------------------------------" | tee -a "$SUMMARY_LOG"
echo "纯符号推导解答数:  $SOLVED_DDAR_ONLY" | tee -a "$SUMMARY_LOG"
echo "使用辅助点解答数:  $SOLVED_WITH_AUX" | tee -a "$SUMMARY_LOG"
echo "未能解答数:        $FAILED" | tee -a "$SUMMARY_LOG"
echo "==================================================" | tee -a "$SUMMARY_LOG"
echo "每道题的详细日志已保存在 $OUTPUT_DIR 目录下。"

