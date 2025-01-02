import numpy as np
import torch
import torch.nn as nn
import random
from math import ceil

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

def generate_full_data(MODULUS: int):
    full_data = []
    for i in range(MODULUS):
        for j in range(MODULUS):
            full_data.append([i,j,(i+j)% MODULUS])
    full_data = np.array(full_data)
    return full_data    

def generate_dataset(MODULUS: int, train_fraction: float, args):
    full_data = generate_full_data(MODULUS)
    np.random.shuffle(full_data)
    train_data = full_data[:int(len(full_data) * train_fraction)]
    test_data = full_data[int(len(full_data) * train_fraction):]
    BATCH_SIZE = 128
    BATCH_NUM = ceil(len(train_data)/BATCH_SIZE)
    train_data = torch.tensor(train_data, dtype=torch.long, device=args.device)
    test_data = torch.tensor(test_data, dtype=torch.long, device=args.device)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    

if __name__ == "__main__":
    set_seed()
    generate_dataset(61,0.5)
