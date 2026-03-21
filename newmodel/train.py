import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm   # 导入进度条库
from tokenizer import GeometryTokenizer
from dataset import GeometryDataset
from model import create_mini_geometry_model

def train():
    # 硬件探测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # 准备基础组件
    data_path = "traindata/synthetic_data_v3.jsonl"
    tokenizer = GeometryTokenizer()
    tokenizer.build_vocab_from_jsonl(data_path)
    vocab_size = len(tokenizer.vocab)
    pad_id = tokenizer.vocab["<pad>"]

    full_dataset = GeometryDataset(data_path, tokenizer, max_length = 196)

    total_size = len(full_dataset)
    val_size = total_size // 10  # 进行模型验证
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"训练集 {train_size} 条, 验证集 {val_size} 条")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 实例化模型并移动到计算设备上
    model = create_mini_geometry_model(vocab_size)
    model.to(device)

    # 定义优化器 AdamW，学习率为 1e-4
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)

    # 学习率调度器：当验证损失不再下降时，降低学习率
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',         # 监控的指标越小越好
        factor=0.5,         # 学习率衰减因子
        patience=2,         # 容忍多少个 epoch 验证损失不下降
        verbose=True        # 打印学习率更新信息
    )

    # ================= 开始训练 =================
    num_epochs = 30                 # 训练轮数
    best_val_loss = float('inf')    # 最佳 Val Loss
    save_dir = "./mini_ag_weights"

    print("\n" + "="*40)
    print("开始训练和验证")
    print("="*40)

    for epoch in range(num_epochs):
        model.train() # 训练模式
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{num_epochs} [Train]",
                          leave=False, ncols=100)  # 进度条不会占用多行
        
        for batch in train_pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = (input_ids != pad_id).long().to(device)

            optimizer.zero_grad()
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss 
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            # 更新进度条显示的实时损失
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)

        # ===================================================

        model.eval() # 评估模式
        total_val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{num_epochs} [Val]",
                        leave=False, ncols=100)
        
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = (input_ids != pad_id).long().to(device)

                outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        avg_val_loss = total_val_loss / len(val_loader)

        # 打印 epoch 汇总信息（独立于进度条，更清晰）
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 根据验证损失调整学习率
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"    发现最佳模型 (Val Loss: {best_val_loss:.4f})")
            model.save_pretrained(save_dir)

    print(f"训练完成。最优 Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()

