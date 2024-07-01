from datahandler import CustomDataset
from models import create_model
from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(dataset, layer_num, layer_links, num_epoch, batch_size, n_cpu, lr, savedir):
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 获得dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)

    # 创建模型
    model = create_model(layer_num, layer_links)
    model = model.to("cuda")

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 创建损失函数
    loss_fn = torch.nn.MSELoss()

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # 早停法
    best_val_loss = float('inf')
    patience = 10
    early_stop_counter = 0

    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = num_epoch  # 设置训练的 epoch 数
    all_losses = []  # 用于存储所有的损失值
    iter_count = 0  # 迭代次数计数器

    for epoch in range(num_epochs):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                x, y = data
                x = x.view(-1, 1).to("cuda")  # 确保数据类型为 float
                y = y.view(-1, 1).to("cuda")  # 确保数据类型为 float
                optimizer.zero_grad()  # 清空梯度

                # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    y_pred = model(x)  # 前向传播
                    loss = loss_fn(y, y_pred)  # 计算损失

                # 反向传播和优化
                scaler.scale(loss).backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                all_losses.append(loss.item())
                iter_count += 1

                # 更新 tqdm 的损失显示
                tepoch.set_postfix(loss=loss.item())
                tepoch.update(1)

        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                x, y = data
                x = x.view(-1, 1).to("cuda")
                y = y.view(-1, 1).to("cuda")
                y_pred = model(x)
                val_loss += loss_fn(y, y_pred).item()
        val_loss /= len(val_loader)
        print(f"Validation Loss after epoch {epoch + 1}: {val_loss}")

        # 更新学习率调度器
        scheduler.step(val_loss)

        # 早停法
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), savedir + f"best_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping")
                break

        # 保存当前 epoch 的模型权重
        torch.save(model.state_dict(), savedir + f"model_epoch_{epoch + 1}.pth")

        # 绘制当前 epoch 的损失曲线
        plt.figure()
        plt.plot(all_losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curve for Epoch')
        plt.savefig(savedir + 'loss_curve_epoch.png')
        plt.close()
