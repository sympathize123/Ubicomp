import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPFeaturizer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, num_layers=3, dropout=0.3):
        super(MLPFeaturizer, self).__init__()
        layers = []
        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden Layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
        # Output Layer (to feature embedding)
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.network(x)

class ResNetFeaturizer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, num_blocks=2, dropout=0.3):
        super(ResNetFeaturizer, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        out = self.output_layer(out)
        return out

class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout):
        super(ResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.linear1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out + residual

class TransformerFeaturizer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128, num_layers=2, nhead=4, dropout=0.1):
        super(TransformerFeaturizer, self).__init__()
        # Feature-token transformer: each feature becomes a token.
        # NOTE: This is standard full attention via nn.TransformerEncoder.
        # Linear attention is NOT implemented here.
        # Embed each scalar feature to hidden_dim and add a learned feature-id embedding.
        self.input_dim = input_dim
        self.feature_embed = nn.Linear(1, hidden_dim)
        self.feature_id_embed = nn.Embedding(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        # x: (batch, input_dim)
        if x.dim() != 2:
            raise ValueError(f"Expected input of shape (batch, features), got {tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {x.shape[1]}")

        # Tokenize features: (B, F) -> (B, F, 1) -> (B, F, H)
        tokens = self.feature_embed(x.unsqueeze(-1))

        # Add learned feature-id embeddings to encode feature identity
        feat_idx = torch.arange(self.input_dim, device=x.device)
        tokens = tokens + self.feature_id_embed(feat_idx).unsqueeze(0)

        # Self-attention across features
        out = self.encoder(tokens)  # (B, F, H)
        out = out.mean(dim=1)       # Pool over features -> (B, H)
        
        return self.output_layer(out)
