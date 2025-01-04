import torch
from math import ceil

def x_plus_y_mod_p_data(p: int, eq_token: int, op_token: int):
    """
    生成 x + y (mod p) 数据集。
    x, y 取值范围为 [0, p)
    eq_token, op_token 用于表示“=”与“+”在 embedding 空间中的 token 索引
    """
    x = torch.arange(0, p)
    y = torch.arange(0, p)
    x, y = torch.cartesian_prod(x, y).T  # 得到所有 (x, y) 组合

    eq = torch.ones_like(x) * eq_token  
    op = torch.ones_like(x) * op_token  

    labels = (x + y) % p  # 标签即 (x+y) mod p

    # 模拟表达式: x + y = ?
    # inputs 形状: [N, 4], 分别是 [x_token, +_token, y_token, =_token]
    inputs = torch.stack([x, op, y, eq], dim=1)
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
    # prime 作为词表中数的范围, eq_token=prime, op_token=prime+1
    inputs, labels = x_plus_y_mod_p_data(
        p=prime,
        eq_token=prime,
        op_token=prime+1
    )
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 为了避免 batch_size 大于总数据的一半，引入此判断
    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=True)

    return train_loader, val_loader