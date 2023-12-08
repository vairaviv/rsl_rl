"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import torch
import torch.nn as nn
from .s4_full import S4Block as S4D


class S4Model(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, d_state=16, batch_first=False, dropout=0.0, keep_states=False):

        super().__init__()

        self.batch_first = batch_first
        self.keep_states = keep_states

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(input_size, hidden_size)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            self.s4_layers.append(S4D(mode='diag', 
                                      init='diag',
                                      dropout=dropout,
                                      final_act='gelu',
                                      d_model=hidden_size,
                                      d_state=d_state))
            self.dropouts.append(nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity())
            
            
    def forward(self, x, h_0=None):
        """
        Input x is shape (B, L, d_input)
        """
        if x.ndim != 3:
            x = x.unsqueeze(dim=1)
            
        if not self.batch_first:
            x = x.transpose(0, 1) # (B, L, d_input)
                
        bath_size, seq_len, _ = x.shape
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        
        if self.keep_states or seq_len == 1:
            # Initialize the hidden states
            h_0 = [s4.default_state(bath_size) for s4 in self.s4_layers] if h_0 is None else h_0
            h_n = []
        else:
            # with no state input or output, can skip the hidden states
            h_0 = None
            h_n = None

        for i, (layer, dropout) in enumerate(zip(self.s4_layers, self.dropouts)):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            h = h_0[i] if h_0 is not None else None

            if seq_len > 1:
                # Apply S4 block: ignore the hidden state input and output
                z, h = layer(z, state=h)
            elif seq_len == 1:
                layer.setup_step()
                z, h = layer.step(z.squeeze(dim=1), h) # squeeze the sequence dimension: (B, 1, d_model) -> (B, d_model)
                z = z.unsqueeze(dim=1) # unsqueeze the sequence dimension: output is (B, 1, d_model)
            else:
                raise ValueError('Sequence length must be at least 1')
            
            # store the hidden states
            if self.keep_states or seq_len == 1:
                h_n.append(h)

            # Dropout on the output of the S4 block
            z = dropout(z)
            
            # residual connection
            x = z + x
        
        if self.keep_states or seq_len == 1:
            h_n = torch.stack(h_n)

        return x, h_n