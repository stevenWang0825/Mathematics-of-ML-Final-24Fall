import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from math import ceil
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import random
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
MODULUS = 97
K = 2
set_seed()
parser = ArgumentParser()
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--epochs", type=int, default = 5000)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--MODULUS", type=int, default=MODULUS)
parser.add_argument("--K", type=int, default=K)
parser.add_argument("--portion", type=float, default=0.3)
# specific to BiRNN
parser.add_argument("--BiRNN_layers", type=int, default=3)
parser.add_argument("--BiRNN_hid_dim", type=int, default=128)
parser.add_argument("--BiRNN_optim", type=str, choices=["RMSprop", "AdamW", "Adam", "SGD","NSGD"], default="AdamW")

args,unknown = parser.parse_known_args()

data = np.load(f"BiRNN_K=2_2e-05_AdamW_97_15000_0.3.npy").T
flag="loss"

if flag=='acc':
    plt.figure(figsize=(10,5))
    plt.plot(data[0], data[1], label="Train acc")
    plt.plot(data[0], data[2], label="Test acc")
    plt.xlabel("Optimization Epochs")
    plt.ylabel("acc")
    plt.title(f"BiRNN acc wrt Epochs for p={args.MODULUS}-K={args.K}_{args.lr}_{args.BiRNN_optim}_using{args.portion}")
    plt.xscale("log")
    plt.legend()
    plt.savefig(f"BBiRNN_K=2_2e-05_AdamW_97_15000_0.3_acc.png")
    plt.show()
elif flag =="loss":
    plt.figure(figsize=(10,5))
    plt.plot(data[0], data[3], label="Train loss")
    plt.plot(data[0], data[4], label="Test loss")
    plt.xlabel("Optimization Epochs")
    plt.ylabel("loss")
    plt.title(f"BiRNN loss wrt Epochs for p={args.MODULUS}-K={args.K}_{args.lr}_{args.BiRNN_optim}_using{args.portion}")
    plt.xscale("log")
    plt.legend()
    plt.savefig(f"BiRNN_K=2_2e-05_AdamW_97_15000_0.3_loss.png")
    plt.show()