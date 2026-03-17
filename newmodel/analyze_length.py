import json
import numpy as np
import matplotlib.pyplot as plt
from tokenizer import GeometryTokenizer

def analyze_sequence_lengths(jsonl_path):
    print("正在加载分词器并构建词表...")
    tokenizer = GeometryTokenizer()
    tokenizer.build_vocab_from_jsonl(jsonl_path)
    
    lengths = []
    
    print("正在计算所有序列的长度...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # 使用 encode 方法将文本转为 token ID 列表
            token_ids = tokenizer.encode(
                premises=data["premises"], 
                target=data["target"], 
                aux_points=data["auxiliary_points"]
            )
            # 记录这条数据的真实长度
            lengths.append(len(token_ids))
            
    # 将列表转化为 numpy 数组以便计算统计指标
    lengths = np.array(lengths)
    
    # === 计算统计学指标 ===
    print("\n" + "="*30)
    print("序列长度统计报告")
    print("="*30)
    print(f"数据总条数: {len(lengths)}")
    print(f"最短序列长度: {np.min(lengths)}")
    print(f"最长序列长度: {np.max(lengths)}")
    print(f"平均序列长度: {np.mean(lengths):.2f}")
    
    p90 = np.percentile(lengths, 90)
    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)
    print(f"90% 的数据长度小于等于: {p90:.0f}")
    print(f"95% 的数据长度小于等于: {p95:.0f}")
    print(f"99% 的数据长度小于等于: {p99:.0f}")
    print("="*30)
    
    # === 绘制直方图直观展示 ===
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 在图上画出 95% 分位数的分界线
    plt.axvline(p95, color='red', linestyle='dashed', linewidth=2, label=f'95th Percentile ({p95:.0f})')
    
    plt.title('Sequence Length Distribution (AlphaGeometry Synthetic Data)')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    print("\n正在生成分布图，请查看弹出的窗口...")
    plt.show()

if __name__ == "__main__":
    data_path = "traindata/synthetic_data.jsonl"
    analyze_sequence_lengths(data_path)

