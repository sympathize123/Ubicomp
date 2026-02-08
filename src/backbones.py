import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPFeaturizer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, dropout=0.3):
        super(MLPFeaturizer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
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
        # Simple projection to hidden_dim then transformer encoder
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*2, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        # x is (batch, input_dim). Transformer expects (batch, seq_len, d_model) or similar.
        # For tabular, this "Transformer" usually treats features as tokens OR simply processes the embedding vector.
        # If we treat the whole input vector as one token (seq_len=1)? That's barely a transformer.
        # FT-Transformer treats each feature as a token.
        # BUT that requires knowing categorical vs continuous and embedding them separately.
        # Since our input is already preprocessed/normalized float vectors, a full column-wise transformer is complex to retrofit without schema.
        # HERE: We will implement a "Row Transformer" or simply an MLP-Mixer style or just apply Self-Attention on the expanded feature dimension?
        # A simple approximation for "Backbone" request on numerical data: 
        # Project to D, reshaped to (Batch, 1, D) -> Transformer -> (Batch, 1, D) -> Flatten
        # This is essentially just self-attention on the latent representation.
        
        x_emb = self.embedding(x) # (Batch, hidden_dim)
        x_emb = x_emb.unsqueeze(1) # (Batch, 1, hidden_dim)
        
        out = self.encoder(x_emb) # (Batch, 1, hidden_dim)
        out = out.squeeze(1) # (Batch, hidden_dim)
        
        out = self.output_layer(out)
        return out
