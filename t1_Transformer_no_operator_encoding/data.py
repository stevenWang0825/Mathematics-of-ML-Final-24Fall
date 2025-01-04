import torch
from math import ceil

def x_plus_y_mod_p_data(p: int):
    """
    生成 x + y (mod p) 数据集。
    x, y 取值范围为 [0, p)
    不再使用 “+” 和 “=” 做 token
    inputs 形状: [N, 2], 分别是 [x_token, y_token]
    labels: (x + y) mod p
    """
    x = torch.arange(0, p)
    y = torch.arange(0, p)
    x, y = torch.cartesian_prod(x, y).T  # 得到所有 (x, y) 组合

    inputs = torch.stack([x, y], dim=1)  # [N, 2]
    labels = (x + y) % p                 # [N]
    return inputs, labels

def get_data(
    prime: int,
    training_fraction: float,
    batch_size: int
):
    """
    返回 DataLoader: 训练集与验证集。
    仅针对 x+y (mod prime)
    """
    inputs, labels = x_plus_y_mod_p_data(p=prime)
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 为了避免 batch_size 大于总数据的一半
    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=True)

    return train_loader, val_loader