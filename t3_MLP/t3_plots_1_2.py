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

    xdrop = [0,0.1,0.2,0.3,0.4,0.5]
    xdecay = [0,0.2,0.4,0.6,0.8,1.0]
    xfrac = [0.2,0.3,0.4,0.5,0.6,0.7]

    xembd = [32,64,96,128]
    xhid = [32,64,128,256]
    xlayer = [2,3,4,5]

    decay = np.load("decay_accs.npy")
    print(decay)
    dropout = np.load("dropout_accs.npy")
    print(dropout)
    fraction = np.load("frac_accs.npy")
    print(fraction)

    embd = np.load("embd_accs.npy")
    print(embd)
    hid = np.load("hid_accs.npy")
    print(hid)
    layer = np.load("layer_accs.npy")
    print(layer)

    fig, ax = plt.subplots(2, 3, figsize=(8, 8 * 0.618))

    # First row, first plot (Dropout)
    ax[0, 0].plot(xdrop, dropout, marker='o', linestyle='-', color='b')
    ax[0, 0].set_title("Dropout")
    ax[0, 0].set_ylim(0., 1)

    # First row, second plot (Weight Decay)
    ax[0, 1].plot(xdecay, decay, marker='o', linestyle='-', color='g')
    ax[0, 1].set_title("Weight Decay")

    # First row, third plot (Fraction)
    ax[0, 2].plot(xfrac, fraction, marker='o', linestyle='-', color='r')
    ax[0, 2].set_title("Fraction")

    # Second row, first plot (Embedding Dim)
    ax[1, 0].plot(xembd, embd, marker='o', linestyle='-', color='b')
    ax[1, 0].set_title("Embedding Dim")
    ax[1, 0].set_ylim(0.8, 1)

    # Second row, second plot (Hidden Dim)
    ax[1, 1].plot(xhid, hid, marker='o', linestyle='-', color='g')
    ax[1, 1].set_title("Hidden Dim")
    ax[1, 1].set_ylim(0.8, 1)

    # Second row, third plot (Layers)
    ax[1, 2].plot(xlayer, layer, marker='o', linestyle='-', color='r')
    ax[1, 2].set_title("Layers")
    ax[1, 2].set_ylim(0.8, 1)

    
    fig.suptitle("Hyperparameters and Regularization")
    plt.tight_layout()
    plt.savefig("t3_plots_1.png",dpi=500)
    plt.show()

    rmsprop = np.load("MLP_binary_59_256_RMSprop_0.0005.npy").T
    adamw = np.load("MLP_binary_59_256_AdamW_0.0005.npy").T
    adam = np.load("MLP_binary_59_256_Adam_0.0005.npy").T
    sgd = np.load("MLP_binary_59_256_SGD_0.0005.npy").T
    nadam = np.load("MLP_binary_59_256_NAdam_0.0005.npy").T

    fig, ax = plt.subplots(2, 2, figsize=(8, 8 * 0.618))
    ax[0,0].plot(rmsprop[0], rmsprop[1],  color='b', label="RMSprop")
    ax[0,0].plot(adamw[0], adamw[1],  color='g', label="AdamW")
    ax[0,0].plot(adam[0], adam[1],  color='r', label="Adam")
    ax[0,0].plot(sgd[0], sgd[1],  color='c', label="SGD")
    ax[0,0].plot(nadam[0], nadam[1],  color='m', label="NAdam")
    ax[0,0].set_title("Train accuracy")
    ax[0,0].set_ylim(0., 1)
    ax[0,0].legend()
    ax[0,0].set_xscale("log")

    ax[0,1].plot(rmsprop[0], rmsprop[2],  color='b', label="RMSprop")
    ax[0,1].plot(adamw[0], adamw[2],  color='g', label="AdamW")
    ax[0,1].plot(adam[0], adam[2],  color='r', label="Adam")
    ax[0,1].plot(sgd[0], sgd[2],  color='c', label="SGD")
    ax[0,1].plot(nadam[0], nadam[2],  color='m', label="NAdam")
    ax[0,1].set_title("Test accuracy")
    ax[0,1].set_ylim(0., 1)
    ax[0,1].legend()
    ax[0,1].set_xscale("log")

    ax[1,0].plot(rmsprop[0], rmsprop[3],  color='b', label="RMSprop")
    ax[1,0].plot(adamw[0], adamw[3],  color='g', label="AdamW")
    ax[1,0].plot(adam[0], adam[3],  color='r', label="Adam")
    ax[1,0].plot(sgd[0], sgd[3],  color='c', label="SGD")
    ax[1,0].plot(nadam[0], nadam[3],  color='m', label="NAdam")
    ax[1,0].set_title("Train loss")
    ax[1,0].legend()
    ax[1,0].set_xscale("log")

    ax[1,1].plot(rmsprop[0], rmsprop[4],  color='b', label="RMSprop")
    ax[1,1].plot(adamw[0], adamw[4],  color='g', label="AdamW")
    ax[1,1].plot(adam[0], adam[4],  color='r', label="Adam")
    ax[1,1].plot(sgd[0], sgd[4],  color='c', label="SGD")
    ax[1,1].plot(nadam[0], nadam[4],  color='m', label="NAdam")
    ax[1,1].set_title("Test loss")
    ax[1,1].legend()
    ax[1,1].set_xscale("log")

    fig.suptitle("Optimizers")
    plt.tight_layout()
    plt.savefig("t3_plots_2.png",dpi=500)
    plt.show()





    


