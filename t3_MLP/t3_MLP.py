import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from math import ceil
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.manifold import TSNE
import os

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return

class MLP(nn.Module):
    def __init__(self, MODULUS:int, num_layers: int, embedding_dim: int, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.embedding = nn.Embedding(MODULUS, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, MODULUS)
        self.dropout = nn.Dropout(self.dropout_rate)
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
        x = self.dropout(x)
        for _ in range(self.num_layers):
            x = F.relu(self.linear2(x))
            x = self.dropout(x)
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
        self.model = MLP(self.MODULUS, 
                         self.args.mlp_layers,
                         self.args.mlp_embd_dim,
                         self.args.mlp_hid_dim,
                         self.args.dropout).to(self.args.device)

    def generate_multiple_dataset(self, MODULUS: int, K: int, train_fraction: float):
        set_seed(42)
        full_data = self.generate_full_multiple_data(MODULUS, K)
        np.random.shuffle(full_data)
        train_data = full_data[:int(len(full_data) * train_fraction)]
        test_data = full_data[int(len(full_data) * train_fraction):]
        # BATCH_SIZE = 256
        BATCH_SIZE = 512
        self.BATCH_SIZE = BATCH_SIZE
        BATCH_NUM = ceil(len(train_data)/BATCH_SIZE)
        self.BATCH_NUM = BATCH_NUM
        train_data = torch.tensor(train_data, dtype=torch.long, device=self.args.device)
        test_data = torch.tensor(test_data, dtype=torch.long, device=self.args.device)
        self.train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    def train_binary(self):
        if self.args.MLP_optim == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.adamw_weight_decay)
        elif self.args.MLP_optim == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.adam_weight_decay)
        elif self.args.MLP_optim == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr)
        elif self.args.MLP_optim == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        elif self.args.MLP_optim == "NSGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, nesterov=True)
        elif self.args.MLP_optim == "NAdam":
            self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.args.lr)
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
        # print(self.rounds_log.shape, self.train_acc_log.shape, self.test_acc_log.shape, self.train_loss_log.shape, self.test_loss_log.shape)
        logs = np.stack([
        np.array(self.rounds_log),
        np.array(self.train_acc_log),
        np.array(self.test_acc_log),
        np.array(self.train_loss_log),
        np.array(self.test_loss_log)
        ], axis=1)
        print(logs.shape)
        os.makedirs("decay_logs", exist_ok=True)
        if self.args.MLP_optim == "AdamW":
            np.save(f"decay_logs/MLP_{self.args.MLP_optim}_{self.args.lr}_{self.args.adamw_weight_decay}.npy", logs)
        elif self.args.MLP_optim == "Adam":
            np.save(f"decay_logs/MLP_{self.args.MLP_optim}_{self.args.lr}_{self.args.adam_weight_decay}.npy", logs)

        # plt.figure(figsize=(6,5))
        # plt.plot(self.rounds_log, self.train_acc_log, label="Train Accuracy")
        # plt.plot(self.rounds_log, self.test_acc_log, label="Test Accuracy")
        # plt.xlabel("Optimization Epochs")
        # plt.ylabel("Accuracy")
        # plt.title(f"MLP Accuracy wrt Epochs-p={self.MODULUS},lr={self.args.lr},Optim={self.args.MLP_optim},layers={self.args.mlp_layers}")
        # plt.xscale("log")
        # plt.legend()
        # plt.savefig(f"MLP_{self.MODULUS}_{self.args.lr}_{self.args.MLP_optim}_{self.args.mlp_layers}-accuracy.png")
        # plt.show()
        pass

    def get_validation_accuracy(self):
        return np.max(self.test_acc_log)
    

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
    parser.add_argument("--MLP_optim", type=str, choices=["RMSprop", "AdamW", "Adam","SGD","NSGD","NAdam"], default="NAdam")
    parser.add_argument("--adamw_weight_decay", type=float, default=1)
    parser.add_argument("--adam_weight_decay", type=float, default=5e-4)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--mlp_embd_dim", type=int, default=32)
    parser.add_argument("--mlp_hid_dim", type=int, default=64)
    args = parser.parse_args()
    # for item in vars(args):
    #     print(item, ":", vars(args)[item])

    dropout_accs = np.zeros(6)
    for i, dropout in enumerate([0,0.1,0.2,0.3,0.4,0.5]):
        args.dropout = dropout
        trainer = MLPTrainer(args)
        raw_data = trainer.generate_full_binary_data(MODULUS)
        trainer.generate_binary_dataset(MODULUS, args.fraction)
        trainer.train_binary()
        dropout_accs[i] = trainer.get_validation_accuracy()
    args.dropout = 0.0
    np.save("dropout_accs.npy", dropout_accs)

    decay_accs = np.zeros(6)
    # torch.save(trainer.model.state_dict(),"trained_model.pth")
    for i, decay in enumerate([0,0.2,0.4,0.6,0.8,1.0]):
        args.adamw_weight_decay = decay
        trainer = MLPTrainer(args)
        raw_data = trainer.generate_full_binary_data(MODULUS)
        trainer.generate_binary_dataset(MODULUS, args.fraction)
        trainer.train_binary()
        decay_accs[i] = trainer.get_validation_accuracy()
    args.adamw_weight_decay = 1.0
    np.save("decay_accs.npy", decay_accs)

    frac_accs = np.zeros(6)
    for i, fraction in enumerate([0.2,0.3,0.4,0.5,0.6,0.7]):
        args.fraction = fraction
        trainer = MLPTrainer(args)
        raw_data = trainer.generate_full_binary_data(MODULUS)
        trainer.generate_binary_dataset(MODULUS, args.fraction)
        trainer.train_binary()
        frac_accs[i] = trainer.get_validation_accuracy()
    args.fraction = 0.5
    np.save("frac_accs.npy", frac_accs)

    hid_accs = np.zeros(4)
    for i, hidden in enumerate([32,64,128,256]):
        args.mlp_hid_dim = hidden
        trainer = MLPTrainer(args)
        raw_data = trainer.generate_full_binary_data(MODULUS)
        trainer.generate_binary_dataset(MODULUS, args.fraction)
        trainer.train_binary()
        hid_accs[i] = trainer.get_validation_accuracy()
    args.mlp_hid_dim = 128
    np.save("hid_accs.npy", hid_accs)

    embd_accs = np.zeros(4)
    for i, embd in enumerate([32,64,96,128]):
        args.mlp_embd_dim = embd
        trainer = MLPTrainer(args)
        raw_data = trainer.generate_full_binary_data(MODULUS)
        trainer.generate_binary_dataset(MODULUS, args.fraction)
        trainer.train_binary()
        embd_accs[i] = trainer.get_validation_accuracy()
    args.mlp_embd_dim = 64
    np.save("embd_accs.npy", embd_accs)

    layer_accs = np.zeros(4)
    for i, layers in enumerate([2,3,4,5]):
        args.mlp_layers = layers
        trainer = MLPTrainer(args)
        raw_data = trainer.generate_full_binary_data(MODULUS)
        trainer.generate_binary_dataset(MODULUS, args.fraction)
        trainer.train_binary()
        layer_accs[i] = trainer.get_validation_accuracy()
    args.mlp_layers = 4
    np.save("layer_accs.npy", layer_accs)

    opts_accs = np.zeros(5)
    opts = ["RMSprop", "AdamW", "Adam","SGD","NAdam"]
    for i, opt in enumerate(opts):
        args.MLP_optim = opt
        trainer = MLPTrainer(args)
        raw_data = trainer.generate_full_binary_data(MODULUS)
        trainer.generate_binary_dataset(MODULUS, args.fraction)
        trainer.train_binary()
        trainer.save_and_plot()
        opts_accs[i] = trainer.get_validation_accuracy()
    args.MLP_optim = "AdamW"
    np.save("opts_accs.npy", opts_accs)

    lr  = [1e-4,5e-4,5e-3,1e-3]
    adamw_decay = [0.1,0.2,0.5,1.0]
    dec_opt = ["AdamW", "Adam"]
    args.MLP_optim = "AdamW"
    decayed_adams_accs = np.zeros((4,4))

    for i, lr_ in enumerate(lr):
        for j, decay in enumerate(adamw_decay):
            args.lr = lr_
            args.adamw_weight_decay = decay
            trainer = MLPTrainer(args)
            raw_data = trainer.generate_full_binary_data(MODULUS)
            trainer.generate_binary_dataset(MODULUS, args.fraction)
            trainer.train_binary()
            trainer.save_and_plot()
            decayed_adams_accs[i,j] = trainer.get_validation_accuracy()

    args.lr = 5e-4
    args.adamw_weight_decay = 5e-4


    


