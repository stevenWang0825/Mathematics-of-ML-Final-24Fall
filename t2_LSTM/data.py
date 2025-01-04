import torch
from math import ceil

def generate_data(prime, training_fraction):
    """
    生成 x + y (mod prime) 数据集
    """
    x = torch.arange(prime)
    y = torch.arange(prime)
    x, y = torch.meshgrid(x, y, indexing='ij')
    x, y = x.flatten(), y.flatten()

    # 将第三个token设为 equals_token
    equals = torch.ones_like(x) * prime
    inputs = torch.stack([x, y, equals], dim=1)
    labels = (x + y) % prime

    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

def get_data_loaders(prime, training_fraction, batch_size):
    """
    返回训练集和验证集的 DataLoader
    """
    train_dataset, val_dataset = generate_data(prime, training_fraction)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
