import torch
from math import ceil
from tqdm import tqdm
import wandb

from data import get_data
from model import Transformer

def main(args: dict):
    wandb.init(project="grokking", config=args)
    config = wandb.config

    device = torch.device(config.device)

    # 定义 Metrics
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("training/accuracy", step_metric='step')
    wandb.define_metric("training/loss", step_metric='step')
    wandb.define_metric("validation/accuracy", step_metric='epoch')
    wandb.define_metric("validation/loss", step_metric='epoch')

    # 获取 DataLoader (改动在 data.py 中)
    train_loader, val_loader = get_data(
        prime=config.prime,
        training_fraction=config.training_fraction,
        batch_size=config.batch_size
    )

    # 初始化模型
    # ★ 这里 num_tokens=prime (词表大小就是 prime)，seq_len=2 (只有 x 和 y)
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=config.prime,  # <-- 改为 prime，而非 prime+2
        seq_len=2                 # <-- 改为 2，而非 4
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

    # 计算 epoch 数
    num_epochs = ceil(config.num_steps / len(train_loader))

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    global_step = 0
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        # 训练
        t_loss, _ = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, 
            epoch, global_step, config.num_steps
        )
        # 训练集评估
        t_loss, t_acc = evaluate(model, train_loader, device, epoch, mode="training")
        train_losses.append(t_loss)
        train_accs.append(t_acc)

        global_step += len(train_loader)
        if global_step >= config.num_steps:
            # 达到最大 step 则提前退出
            v_loss, v_acc = evaluate(model, val_loader, device, epoch, mode="validation")
            val_losses.append(v_loss)
            val_accs.append(v_acc)
            break

        # 验证集评估
        v_loss, v_acc = evaluate(model, val_loader, device, epoch, mode="validation")
        val_losses.append(v_loss)
        val_accs.append(v_acc)

    # 保存 metrics
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

        total_loss += loss.item()

        if step >= max_steps:
            break

    return total_loss / count_batches, step - start_step

def evaluate(model, data_loader, device, epoch, mode="validation"):
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