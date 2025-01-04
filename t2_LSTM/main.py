import torch
import torch.nn as nn
import numpy as np

from data import get_data_loaders
from model import LSTMModel
from train import train_model

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 超参数
    prime = 47
    training_fraction = 0.3
    batch_size = 128
    num_epochs = 100000
    learning_rate = 1e-3
    weight_decay = 1e-1

    # 获取 DataLoader
    train_loader, val_loader = get_data_loaders(prime, training_fraction, batch_size)

    # 构建 LSTM 模型
    # d_vocab = prime + 1, output_dim = prime
    model = LSTMModel(
        d_vocab=prime + 1,
        d_model=128,
        d_hidden=256,
        num_layers=2,
        output_dim=prime
    ).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98)
    )
    criterion = nn.CrossEntropyLoss()

    # 训练
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, optimizer, criterion, device, num_epochs
    )

    # 保存训练结果以便 plot_metrics 函数读取
    results_dict = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs
    }
    torch.save(results_dict, "metrics.pt")
    print("Training finished. Saved metrics to metrics.pt.")

if __name__ == "__main__":
    main()