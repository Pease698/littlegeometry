import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tokenizer import GeometryTokenizer
from dataset import GeometryDataset
from model import create_mini_geometry_model

def train():
    # 硬件探测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # 准备基础组件
    data_path = "traindata/synthetic_data.jsonl"
    tokenizer = GeometryTokenizer()
    tokenizer.build_vocab_from_jsonl(data_path)
    vocab_size = len(tokenizer.vocab)
    pad_id = tokenizer.vocab["<pad>"]

    full_dataset = GeometryDataset(data_path, tokenizer, max_length = 160)

    total_size = len(full_dataset)
    val_size = 40  # 进行模型验证
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"训练集 {train_size} 条, 验证集 {val_size} 条")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 实例化模型并移动到计算设备上
    model = create_mini_geometry_model(vocab_size)
    model.to(device)

    # 定义优化器 AdamW，学习率为 5e-4
    optimizer = AdamW(model.parameters(), lr = 3e-4)

    # ================= 开始训练 =================
    num_epochs = 30                 # 训练轮数
    patience = 3                    # 容忍度
    patience_counter = 0            # 计数器
    best_val_loss = float('inf')    # 最佳 Val Loss
    save_dir = "./mini_ag_weights"

    print("\n" + "="*40)
    print("开始训练和验证")
    print("="*40)

    for epoch in range(num_epochs):
        model.train() # 训练模式
        total_train_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = (input_ids != pad_id).long().to(device)

            optimizer.zero_grad()
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss 
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # ===================================================

        model.eval() # 评估模式
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = (input_ids != pad_id).long().to(device)

                outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # 计数器清零
            
            print(f"    发现最佳模型 (Val Loss: {best_val_loss:.4f})")
            model.save_pretrained(save_dir)
        else:
            patience_counter += 1
            print(f"    验证集 Loss 未下降，早停警告: {patience_counter}/{patience}")
            
            # 判断是否耗尽了容忍度
            if patience_counter >= patience:
                print(f"训练结束。最优 Val Loss 定格在: {best_val_loss:.4f}")
                break


if __name__ == "__main__":
    train()

