
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    """
    LSTM-based encoder for 1D K-line sequences.
    Input: (batch, seq_len, input_dim)
    Output: (batch, embedding_dim)
    """
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, embedding_dim=256, prediction_dim=1, num_classes=None):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Bi-LSTM hidden dim is hidden_dim * 2
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

        # Use num_classes if provided, otherwise use prediction_dim
        output_dim = num_classes if num_classes is not None else prediction_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state (concatenate forward and backward)
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)
        last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        embedding = self.embedding_head(last_hidden)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        prediction = self.prediction_head(last_hidden)
        
        return embedding, prediction

class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1DEncoder(nn.Module):
    """
    1D ResNet encoder for 1D K-line sequences.
    Input: (batch, input_dim, seq_len)
    Output: (batch, embedding_dim)
    """
    def __init__(self, input_dim=5, layers=[2, 2, 2, 2], embedding_dim=256, prediction_dim=1, num_classes=None):
        super(ResNet1DEncoder, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.embedding_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

        # Use num_classes if provided, otherwise use prediction_dim
        output_dim = num_classes if num_classes is not None else prediction_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNet1DBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNet1DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, input_dim, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        embedding = self.embedding_head(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        prediction = self.prediction_head(x)
        
        return embedding, prediction

from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    """
    Dataset for 1D sequences with self-supervised augmentation.
    """
    def __init__(self, sequences, labels=None, augment=True):
        self.sequences = sequences # List of (seq_len, 5)
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def _apply_augmentation(self, seq):
        # seq: (seq_len, 5)
        augmented = seq.copy()
        # 1. Add small Gaussian noise
        if np.random.random() > 0.5:
            # Financial data is sensitive, use very small noise
            noise = np.random.normal(0, 0.001, augmented.shape).astype(np.float32)
            augmented += noise
        
        # 2. Scaling
        if np.random.random() > 0.5:
            scaling_factor = np.random.uniform(0.99, 1.01)
            augmented *= scaling_factor
            
        return augmented

    def __getitem__(self, idx):
        # Anchor
        anchor = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32) if self.labels is not None else torch.tensor(0.0)
        
        # Positive: augmented version
        pos_seq = self._apply_augmentation(self.sequences[idx]) if self.augment else self.sequences[idx]
        positive = torch.tensor(pos_seq, dtype=torch.float32)
        
        # Negative: random selection
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = np.random.randint(0, len(self.sequences))
        negative = torch.tensor(self.sequences[neg_idx], dtype=torch.float32)
        
        return anchor, positive, negative, label
