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

class MLP(nn.Module):
    def __init__(self, MODULUS:int, num_layers: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(MODULUS, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, MODULUS)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)

    def forward(self, x1,x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x = torch.concat([x1,x2], dim=1)
        x = F.relu(self.linear1(x))
        for _ in range(self.num_layers):
            x = F.relu(self.linear2(x))
        x = self.output(x)
        return x
    
class MultipleMLP(nn.Module):
    def __init__(self, MODULUS:int, num_layers: int, embedding_dim: int, hidden_dim, K: int):
        super().__init__()
        set_seed(42)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(MODULUS, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * K, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, MODULUS)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
    
    def forward(self, *input):
        embeddings = [self.embedding(x) for x in input]
        x = torch.cat(embeddings, dim=1) # embedding_dim * K
        x = F.relu(self.linear1(x))
        for _ in range(self.num_layers):
            x = F.relu(self.linear2(x))
        x = self.output(x)
        return x


class MLPTrainer:
    def __init__(self, args):
        super(MLPTrainer,self).__init__()
        set_seed(42)
        self.args = args
        self.rounds_log = []
        self.train_acc_log = []
        self.test_acc_log = []
        self.train_loss_log = []
        self.test_loss_log = []

    @staticmethod
    def generate_full_binary_data(MODULUS: int):
        set_seed(42)
        full_data = []
        for i in range(MODULUS):
            for j in range(MODULUS):
                full_data.append([i,j,(i+j)% MODULUS])
        full_data = np.array(full_data)
        return full_data  
    
    @staticmethod
    def generate_full_multiple_data(MODULUS: int, K :int):
        set_seed(42)
        ranges = [range(MODULUS)] * K
        combinations = np.array(np.meshgrid(*ranges)).T.reshape(-1, K)
        modulo_results = combinations.sum(axis=1) % MODULUS
        full_data = np.hstack((combinations, modulo_results.reshape(-1, 1))).shape
        return full_data

    def generate_binary_dataset(self, MODULUS: int, train_fraction: float):
        set_seed(42)
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
        self.model = MLP(self.MODULUS, self.args.mlp_layers, self.args.mlp_embd_dim, self.args.mlp_hid_dim,).to(self.args.device)

    def generate_multiple_dataset(self, MODULUS: int, K: int, train_fraction: float):
        set_seed(42)
        full_data = self.generate_full_multiple_data(MODULUS, K)
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

    def train_binary(self):
        
        if self.args.MLP_optim == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1)
        elif self.args.MLP_optim == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.MLP_optim == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr)
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
                x1, x2, y = data[:,0], data[:,1], data[:,2]
                y_pred = self.model(x1, x2)
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
                    x1, x2, y = data[:,0], data[:,1], data[:,2]
                    y_pred = self.model(x1, x2)
                    # print(y_pred.shape, y.shape)
                    loss = self.criterion(y_pred, y)
                    test_loss += loss.item()
                    test_acc += (y_pred.argmax(dim=1) == y).sum().item()
                test_loss /= (len(self.test_dataloader)*self.BATCH_SIZE)
                test_acc /= (len(self.test_dataloader)*self.BATCH_SIZE)
                self.test_loss_log.append(test_loss)
                self.test_acc_log.append(test_acc)
            if epoch % 2000 == 0:
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
        np.save(f"MLP_binary_{self.MODULUS}_{self.BATCH_SIZE}_{self.args.MLP_optim}_{self.args.lr}.npy", logs)

        plt.figure(figsize=(10,5))
        plt.plot(self.rounds_log, self.train_acc_log, label="Train Accuracy")
        plt.plot(self.rounds_log, self.test_acc_log, label="Test Accuracy")
        plt.xlabel("Optimization Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"MLP Accuracy wrt Epochs-p={self.MODULUS},lr={self.args.lr},Optim={self.args.MLP_optim},layers={self.args.mlp_layers}")
        plt.xscale("log")
        plt.legend()
        plt.savefig(f"MLP_{self.MODULUS}_{self.args.lr}_{self.args.MLP_optim}_{self.args.mlp_layers}-accuracy.png")
        plt.show()
        pass
    

if __name__=="__main__":
    set_seed(42)
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default= 1e4)
    parser.add_argument("--lr", type=float, default=5e-4)
    # specific to MLP
    parser.add_argument("--mlp_layers", type=int, default=4)
    parser.add_argument("--mlp_embd_dim", type=int, default=64)
    parser.add_argument("--mlp_hid_dim", type=int, default=128)
    parser.add_argument("--MLP_optim", type=str, choices=["RMSprop", "AdamW", "Adam"], default="AdamW")
    
    args = parser.parse_args()

    trainer = MLPTrainer(args)
    raw_data = trainer.generate_full_binary_data(59)
    trainer.generate_binary_dataset(59, 0.5)
    embeddings = trainer.model.embedding(torch.tensor(np.arange(59), dtype=torch.long, device=args.device))
    matrix = embeddings.detach().cpu().numpy()
    # print(trainer.model.embedding(torch.tensor(np.arange(41), dtype=torch.long, device=args.device)).shape)
    # print(len(trainer.train_dataloader))
    trainer.train_binary()
    embeddings = trainer.model.embedding(torch.tensor(np.arange(59), dtype=torch.long, device=args.device))
    matrix = embeddings.detach().cpu().numpy()
    trainer.save_and_plot()

