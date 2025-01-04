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

class BiRNN(nn.Module):
    def __init__(self, K, MODULUS, hidden_size, num_layers):
        super(BiRNN, self).__init__()
        # Bidirectional RNN layer
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers = num_layers, bidirectional=True, batch_first=True)
        # Fully connected layer for output
        self.fc = nn.Linear(2 * hidden_size, MODULUS)  # 2 * hidden_size because it's bidirectional

    def forward(self, x):
        # Forward pass through RNN
        out, _ = self.rnn(x)
        # Take the output from the last time step
        out_last = out[:, -1, :]
        # Pass it through the fully connected layer
        out_final = self.fc(out_last)
        return out_final
    
class BiRNNTrainer:
    def __init__(self, args):
        super(BiRNNTrainer,self).__init__()
        self.args = args
        self.rounds_log = []
        self.train_acc_log = []
        self.test_acc_log = []
        self.train_loss_log = []
        self.test_loss_log = []

    @staticmethod
    def generate_full_binary_data(MODULUS: int):
        full_data = []
        for i in range(MODULUS):
            for j in range(MODULUS):
                full_data.append([i,j,(i+j)% MODULUS])
        full_data = np.array(full_data)
        return full_data  
    
    @staticmethod
    def generate_full_multiple_data(MODULUS: int, K :int):
        ranges = [range(MODULUS)] * K
        combinations = np.array(np.meshgrid(*ranges)).T.reshape(-1, K)
        modulo_results = combinations.sum(axis=1) % MODULUS
        full_data = np.hstack((combinations, modulo_results.reshape(-1, 1)))
        return full_data

    def generate_binary_dataset(self, MODULUS: int, train_fraction: float):
        self.MODULUS = MODULUS
        full_data = self.generate_full_binary_data(MODULUS)
        np.random.shuffle(full_data)
        train_data = full_data[:int(len(full_data) * train_fraction)]
        test_data = full_data[int(len(full_data) * train_fraction):]
        BATCH_SIZE = 256
        self.BATCH_SIZE = BATCH_SIZE
        BATCH_NUM = ceil(len(train_data)/BATCH_SIZE)
        self.BATCH_NUM = BATCH_NUM
        train_data = torch.tensor(train_data, dtype=torch.long, device=self.args.device)
        test_data = torch.tensor(test_data, dtype=torch.long, device=self.args.device)
        self.train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    def generate_multiple_dataset(self, MODULUS: int, K: int, train_fraction: float):
        full_data = self.generate_full_multiple_data(MODULUS, K)
        np.random.shuffle(full_data)
        full_data = torch.tensor(full_data, dtype=torch.float32, device=self.args.device)
        full_data = full_data.unsqueeze(-1)
        train_data = full_data[:int(len(full_data) * train_fraction)]
        test_data = full_data[int(len(full_data) * train_fraction):]
        BATCH_SIZE = 512
        self.BATCH_SIZE = BATCH_SIZE
        BATCH_NUM = ceil(len(train_data)/BATCH_SIZE)
        self.BATCH_NUM = BATCH_NUM
        self.train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    def train(self):
        self.model = BiRNN(self.args.K, self.args.MODULUS,self.args.BiRNN_hid_dim, self.args.BiRNN_layers).to(self.args.device)
        if self.args.BiRNN_optim == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr,weight_decay=1)
        elif self.args.BiRNN_optim == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.BiRNN_optim == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr)
        elif self.args.BiRNN_optim == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        elif self.args.BiRNN_optim == "NSGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, nesterov=True)
        else:
            raise ValueError("Invalid Optimizer")

        self.criterion = nn.CrossEntropyLoss()
        num_epochs = int(self.args.epochs)
        for epoch in tqdm(range(num_epochs),desc="Epoch"):
            # training procedure
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            for i, data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                x, y = data[:, :-1,:], data[:,-1,:].squeeze(-1).long()
                y_pred = self.model(x)
                y = y.long()
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_acc += (y_pred.argmax(dim=1) == y).sum().item()

            self.rounds_log.append(epoch * self.BATCH_NUM)
            train_loss /=(len(self.train_dataloader)*self.BATCH_SIZE)
            train_acc /= (len(self.train_dataloader)*self.BATCH_SIZE)
            self.train_loss_log.append(train_loss)
            self.train_acc_log.append(train_acc)
            # testing procedure
            self.model.eval()
            test_acc = 0.0
            test_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    x, y = data[:, :-1,:], data[:,-1,:].squeeze(-1).long()
                    y_pred = self.model(x)
                    y = y.long()
                    loss = self.criterion(y_pred, y)
                    test_loss += loss.item()
                    test_acc += (y_pred.argmax(dim=1) == y).sum().item()
                test_loss /= (len(self.test_dataloader)*self.BATCH_SIZE)
                test_acc /= (len(self.test_dataloader)*self.BATCH_SIZE)
                self.test_loss_log.append(test_loss)
                self.test_acc_log.append(test_acc)
            if epoch % 20 == 0 and epoch < 100:
                print(f"Epoch {epoch} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")
                print(f"Epoch {epoch} Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")
            elif epoch % 100 == 0 and epoch >= 100:
                print(f"Epoch {epoch} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")
                print(f"Epoch {epoch} Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")
        print("Training Finished")
        return
    
    def save_and_plot(self):
        logs = np.stack([
        np.array(self.rounds_log),
        np.array(self.train_acc_log),
        np.array(self.test_acc_log),
        np.array(self.train_loss_log),
        np.array(self.test_loss_log)
        ], axis=1)
        print(logs.shape)
        np.save(f"BiRNN_K={self.args.K}_{self.args.lr}_{self.args.BiRNN_optim}_{self.args.MODULUS}_{self.args.epochs}_{self.args.portion}.npy", logs)

        plt.figure(figsize=(10,5))
        plt.plot(self.rounds_log, self.train_acc_log, label="Train Accuracy")
        plt.plot(self.rounds_log, self.test_acc_log, label="Test Accuracy")
        plt.xlabel("Optimization Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"BiRNN Acc wrt Epochs for p={self.args.MODULUS}-K={self.args.K}_{self.args.lr}_{self.args.BiRNN_optim}_using{self.args.portion}")
        plt.xscale("log")
        plt.legend()
        plt.savefig(f"BiRNN_{self.args.BiRNN_optim}_{self.args.MODULUS}_K={self.args.K}_{self.args.lr}_{self.args.epochs}_{self.args.portion}_accuracy.png")
        plt.show()
        pass

if __name__ == "__main__":
    MODULUS = 97
    K = 2
    set_seed()
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default = 15000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--MODULUS", type=int, default=MODULUS)
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--portion", type=float, default=0.3)
    # specific to BiRNN
    parser.add_argument("--BiRNN_layers", type=int, default=3)
    parser.add_argument("--BiRNN_hid_dim", type=int, default=128)
    parser.add_argument("--BiRNN_optim", type=str, choices=["RMSprop", "AdamW", "Adam", "SGD","NSGD"], default="AdamW")
    
    args = parser.parse_args()

    trainer = BiRNNTrainer(args)
    trainer.generate_multiple_dataset(MODULUS, K, args.portion)
    trainer.train()
    trainer.save_and_plot()

    