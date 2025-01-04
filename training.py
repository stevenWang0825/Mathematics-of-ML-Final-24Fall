import torch
from math import ceil
from tqdm import tqdm
import wandb

from data import get_data
from model import Transformer

def main(args: dict):
    # 初始化 wandb （可根据需要决定是否去掉）
    wandb.init(project="grokking", config=args)
    config = wandb.config

    device = torch.device(config.device)

    # 定义要记录的 metric
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("training/accuracy", step_metric='step')
    wandb.define_metric("training/loss", step_metric='step')
    wandb.define_metric("validation/accuracy", step_metric='epoch')
    wandb.define_metric("validation/loss", step_metric='epoch')

    # 获取 DataLoader
    train_loader, val_loader = get_data(
        prime=config.prime,
        training_fraction=config.training_fraction,
        batch_size=config.batch_size
    )

    # 初始化模型
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=config.prime + 2,  # x ∈ [0, prime), op=prime+1, eq=prime
        seq_len=4  # 我们的输入长度是 4 (x, +, y, =)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=9
    )

    # 计算总的 epoch 数
    num_epochs = ceil(config.num_steps / len(train_loader))

    # 用于本地画图的记录
    train_losses = []
    train_accs   = []
    val_losses   = []
    val_accs     = []

    global_step = 0

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        # 训练一个 epoch
        t_loss, _ = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, 
            epoch, global_step, config.num_steps
        )
        # 重新评估整个训练集上的 loss 和 accuracy
        t_loss, t_acc = evaluate(model, train_loader, device, epoch, mode="training")
        train_losses.append(t_loss)
        train_accs.append(t_acc)

        global_step += len(train_loader)  # 累加步数
        if global_step >= config.num_steps:
            # 如果已经到达指定的更新步数，就结束
            v_loss, v_acc = evaluate(model, val_loader, device, epoch, mode="validation")
            val_losses.append(v_loss)
            val_accs.append(v_acc)
            break

        # 每个 epoch 结束，做一次验证
        v_loss, v_acc = evaluate(model, val_loader, device, epoch, mode="validation")
        val_losses.append(v_loss)
        val_accs.append(v_acc)

    # 训练结束后，为了后续画图，把结果保存到本地文件
    torch.save({
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs
    }, "metrics.pt")

def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, start_step, max_steps):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    count_batches = 0

    step = start_step

    for batch in train_loader:
        step += 1
        count_batches += 1

        inputs, labels = (t.to(device) for t in batch)

        optimizer.zero_grad()
        logits = model(inputs)[-1, :, :]  # 取最后一个 token 的输出
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 累计
        total_loss += loss.item()

        if step >= max_steps:
            break

    return total_loss / count_batches, step - start_step

def evaluate(model, data_loader, device, epoch, mode="validation"):
    """
    用整个数据集计算 loss 和 accuracy
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = (t.to(device) for t in batch)
            logits = model(inputs)[-1, :, :]  # 取最后一个 token 的输出
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    # wandb 记录
    if mode == "validation":
        wandb.log({
            "validation/loss": avg_loss,
            "validation/accuracy": accuracy,
            "epoch": epoch
        }, commit=True)
    elif mode == "training":
        wandb.log({
            "training/loss": avg_loss,
            "training/accuracy": accuracy,
            "epoch": epoch
        }, commit=True)

    return avg_loss, accuracy