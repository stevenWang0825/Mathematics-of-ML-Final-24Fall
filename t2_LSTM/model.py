import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, d_vocab, d_model, d_hidden, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(d_vocab, d_model)
        self.lstm = nn.LSTM(d_model, d_hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(d_hidden, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, d_model]
        output, (hn, cn) = self.lstm(embedded)  # [batch_size, seq_len, d_hidden]
        logits = self.fc(output[:, -1, :])  # 只取最后一个时间步
        return logits