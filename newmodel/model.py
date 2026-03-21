import torch
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizer import GeometryTokenizer # 导入你的分词器

def create_mini_geometry_model(vocab_size):
    """
    创建一个微型的 Transformer 模型用于几何辅助点预测。
    """
    print(f"词表大小设为: {vocab_size}")
    
    # 定义模型配置
    config = GPT2Config(
        vocab_size = vocab_size,
        n_positions = 256,       # 最大序列长度
        n_embd = 256,            # 词向量的维度
        n_layer = 16,             # Transformer 的层数
        n_head = 16,              # 多头注意力的头数
        bos_token_id = 1,        # 序列开头，即 <bos> ID
        eos_token_id = 2,        # 序列结尾，即 <eos> ID
        pad_token_id = 0         # 序列填充，即 <pad> ID
    )
    
    # 实例化模型
    model = GPT2LMHeadModel(config)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"微型 AlphaGeometry 模型创建成功")
    print(f"总参数量: {total_params / 1e6:.2f} M (百万)")
    
    return model

if __name__ == "__main__":
    # 初始化分词器获取词表大小
    tokenizer = GeometryTokenizer()
    data_path = "traindata/synthetic_data.jsonl" 
    tokenizer.build_vocab_from_jsonl(data_path)
    
    vocab_size = len(tokenizer.vocab)
    
    # 获取特殊字符的真实 ID，更新配置中的默认值
    bos_id = tokenizer.vocab["<bos>"]
    eos_id = tokenizer.vocab["<eos>"]
    pad_id = tokenizer.vocab["<pad>"]
    
    # 创建模型
    model = create_mini_geometry_model(vocab_size)
    
    # 测试一次前向传播 (Forward Pass)
    print("\n=== 测试前向传播 ===")
    dummy_input = torch.zeros((2, 10), dtype = torch.long) 
    
    outputs = model(input_ids = dummy_input)
    
    # logits 输出为 [Batch_size, Sequence_length, Vocab_size]
    print(f"输入张量形状: {dummy_input.shape}")
    print(f"输出 Logits 形状: {outputs.logits.shape}")

