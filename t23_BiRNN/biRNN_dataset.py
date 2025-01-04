import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BiDirectionalRNN(nn.Module):
    def __init__(self, MODULUS: int, num_layers: int, embedding_dim: int, hidden_dim: int, K: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(MODULUS, embedding_dim)
        
        # Forward and backward RNN linear layers
        self.forward_rnn = nn.Linear(embedding_dim, hidden_dim)
        self.backward_rnn = nn.Linear(embedding_dim, hidden_dim)
        
        # Linear layers after concatenating forward and backward outputs
        self.linear1 = nn.Linear(2 * hidden_dim * K, hidden_dim)
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
        # Embed input sequences
        embeddings = [self.embedding(x) for x in input]  # Each x is (batch_size, seq_len)
        
        # Forward RNN
        forward_outputs = []
        for embedding in embeddings:
            h_forward = torch.zeros(embedding.size(0), self.hidden_dim, device=embedding.device)
            for t in range(embedding.size(1)):
                h_forward = F.relu(self.forward_rnn(embedding[:, t, :]) + h_forward)
                forward_outputs.append(h_forward)

        # Backward RNN
        backward_outputs = []
        for embedding in embeddings:
            h_backward = torch.zeros(embedding.size(0), self.hidden_dim, device=embedding.device)
            for t in reversed(range(embedding.size(1))):
                h_backward = F.relu(self.backward_rnn(embedding[:, t, :]) + h_backward)
                backward_outputs.append(h_backward)

        # Concatenate forward and backward outputs
        combined_outputs = [
            torch.cat((f_out, b_out), dim=1) for f_out, b_out in zip(forward_outputs, backward_outputs)
        ]
        x = torch.cat(combined_outputs, dim=1)  # Concatenate across K inputs

        # Pass through fully connected layers
        x = F.relu(self.linear1(x))
        for _ in range(self.num_layers):
            x = F.relu(self.linear2(x))
        x = self.output(x)
        print(f"Embedding shape: {[e.shape for e in embeddings]}")  # After embedding
        print(f"Forward outputs shape: {len(forward_outputs)}, {[f.shape for f in forward_outputs]}")
        print(f"Backward outputs shape: {len(backward_outputs)}, {[b.shape for b in backward_outputs]}")
        print(f"Combined outputs shape: {len(combined_outputs)}, {[c.shape for c in combined_outputs]}")
        print(f"x shape after concatenation: {x.shape}")
        return x
    
def generate_full_multiple_data(MODULUS: int, K :int):
    ranges = [range(MODULUS)] * K
    combinations = np.array(np.meshgrid(*ranges)).T.reshape(-1, K)
    modulo_results = combinations.sum(axis=1) % MODULUS
    full_data = np.hstack((combinations, modulo_results.reshape(-1, 1)))
    return full_data
    
if __name__=="__main__":
    full_data = generate_full_multiple_data(10, 3)
    print(full_data)
    model = BiDirectionalRNN(MODULUS=10, num_layers=2, embedding_dim=5, hidden_dim=3, K=3)
    # x1=5
    # x2=6
    # x3=7
    # output = model(x1, x2, x3)
    # print(output.shape)