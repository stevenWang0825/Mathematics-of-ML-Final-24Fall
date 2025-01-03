import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from math import ceil
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.manifold import TSNE

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return

if __name__=="__main__":
    set_seed(42)
    MODULUS = 59
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default= 1e4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fraction", type=float, choices=[0.2,0.3,0.4,0.5,0.6,0.7], default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    # specific to MLP
    parser.add_argument("--MLP_optim", type=str, choices=["RMSprop", "AdamW", "Adam","SGD","NSGD"], default="AdamW")
    parser.add_argument("--adamw_weight_decay", type=float, default=1)
    parser.add_argument("--mlp_layers", type=int, default=4)
    parser.add_argument("--mlp_embd_dim", type=int, default=64)
    parser.add_argument("--mlp_hid_dim", type=int, default=128)
    args = parser.parse_args()

    lr = [0.0001,0.0005,0.005,0.001]
    decay = [0.0001,0.0005,0.005,0.001]
    lr_abb = ['1e-4','5e-4','5e-3','1e-3']
    decay_abb = ['1e-4','5e-4','5e-3','1e-3']
    fig, ax = plt.subplots(4, 8, figsize=(16, 5))
    for i in range(len(lr)):
        for j in range(len(decay)):
            data = np.load(f"decay_logs/MLP_Adam_{lr[i]}_{decay[j]}.npy").T
            ax[i,j].plot(data[0], data[1],  label='Train acc')
            ax[i,j].plot(data[0], data[2],  label='Test acc')
            ax[i,j].set_title(f"({lr_abb[i]},{decay_abb[j]})")
            ax[i,j].set_xscale('log')
            ax[i,j].set_xticks([])
            ax[i,j+4].plot(data[0], data[3],  label='Train loss')
            ax[i,j+4].plot(data[0], data[4],  label='Test loss')
            ax[i,j+4].set_xscale('log')
            ax[i,j+4].set_xticks([])
            ax[i,j+4].set_title(f"({lr_abb[i]},{decay_abb[j]})")
            
    fig.suptitle("Adam (lr,weight_decay) Grid Search")
    plt.tight_layout()
    plt.savefig("Adam_grid_search.png", dpi=500)
    plt.show()
