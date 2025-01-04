import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from math import ceil
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import random
import os

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

if __name__ == "__main__":
    MODULUS = 23
    K = 3
    set_seed()
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default = 5000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--MODULUS", type=int, default=MODULUS)
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--portion", type=float, default=0.5)
    # specific to BiRNN
    parser.add_argument("--BiRNN_layers", type=int, default=3)
    parser.add_argument("--BiRNN_hid_dim", type=int, default=32)
    parser.add_argument("--BiRNN_optim", type=str, choices=["RMSprop", "AdamW", "Adam", "SGD","NSGD"], default="AdamW")
    
    args = parser.parse_args()

    lr = [1e-5, 5e-5, 1e-4, 1e-3]
    lr_abb = ["1e-5", "5e-5", "1e-4", "1e-3"]
    fig, ax = plt.subplots(2, 8, figsize=(12,3))
    for i, l in enumerate(lr):
        data = np.load(f"lr_logs/BiRNN_{l}_{args.BiRNN_optim}.npy").T
        ax[0,i].plot(data[0], data[1], label=f"train acc")
        ax[0,i].set_xscale('log')
        ax[0,i].plot(data[0], data[2], label=f"test acc")
        ax[0,i].set_title(f"({lr_abb[i]},1)")
        if i!=0:
            ax[0,i].set_yticks([])
        # ax[0,i].legend()
        ax[1,i].plot(data[0], data[3], label=f"train loss")
        ax[1,i].set_xscale('log')
        ax[1,i].plot(data[0], data[4], label=f"test loss")
        if i!=0:
            ax[1,i].set_yticks([])
        # ax[1,i].legend()

    decay = [0,0.25,0.75,1]
    for j, d in enumerate(decay):
        data = np.load(f"decay_logs/BiRNN_{1e-05}_{d}.npy").T
        ax[0,j+4].plot(data[0], data[1], label=f"train acc")
        ax[0,j+4].set_xscale('log')
        ax[0,j+4].plot(data[0], data[2], label=f"test acc")
        ax[0,j+4].set_title(f"(1e-05,{d})")
        ax[0,j+4].set_yticks([])
        # ax[0,j+4].legend()
        ax[1,j+4].plot(data[0], data[3], label=f"train loss")
        ax[1,j+4].set_xscale('log')
        ax[1,j+4].plot(data[0], data[4], label=f"test loss")
        ax[1,j+4].set_yticks([])
        # ax[1,j+4].legend()
    fig.suptitle(f"BiRNN (lr,weight_decay) partial search")
    plt.tight_layout()
    plt.savefig(f"BiRNN_partial_search.png",dpi=500)
    plt.show()