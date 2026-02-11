import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, num_classes=2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x: (B, Seq_Len, Features)
        
        # Out: (B, Seq, Hidden), (h_n, c_n)
        out, (h_n, c_n) = self.lstm(x)
        
        # Take the last time step hidden state
        # h_n schema: (num_layers, batch, hidden_size)
        last_hidden = out[:, -1, :] # (B, Hidden)
        
        embedding = self.fc_head(last_hidden) #(B, 32)
        logits = self.classifier(embedding) #(B, 2)
        
        return logits, embedding
