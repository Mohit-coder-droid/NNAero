import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from nnaero.models.base import *

'''https://arxiv.org/abs/2004.12585'''
class BN_Layer(nn.Module):
    def __init__(self, dim_z, tau=0.5, mu=True):
        super(BN_Layer, self).__init__()
        self.dim_z = dim_z

        self.tau = torch.tensor(tau)  # tau: float in range (0,1)
        self.theta = torch.tensor(0.5, requires_grad=True)

        self.gamma1 = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.theta))  # for mu
        self.gamma2 = torch.sqrt((1 - self.tau) * torch.sigmoid((-1) * self.theta))  # for var

        self.bn = nn.BatchNorm1d(dim_z)
        self.bn.bias.requires_grad = False
        self.bn.weight.requires_grad = True

        if mu:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma1)
        else:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma2)

    def forward(self, x):  # x:(batch_size,dim_z)
        x = self.bn(x)
        return x

class VAE(nn.Module):
    def __init__(
        self, 
        feature_size: int = 257, 
        latent_size: int = 32,
        encoder_config: list = None,
        decoder_config: list = None
    ):
        super(VAE, self).__init__()
        self.feature_size = feature_size
        
        if encoder_config is None:
            encoder_config = [2048, "relu", 128, "tanh"]
        
        if decoder_config is None:
            decoder_config = [2048, "relu", 128, "relu", feature_size, "tanh"]

        # --- 1. Build Encoder Body ---
        self.encoder_body = BaseNetwork(feature_size, encoder_config)
        
        # The VAE "Heads" (Mu and LogVar) are attached to the end of the encoder body
        last_hidden_dim = self.encoder_body.output_dim
        self.fc_mu = nn.Linear(last_hidden_dim, latent_size)
        self.fc_logvar = nn.Linear(last_hidden_dim, latent_size)

        # --- 2. Build Decoder ---
        self.decoder = BaseNetwork(latent_size, decoder_config)

    def encode(self, x):
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = x.reshape(-1, self.feature_size)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VQ_VAE(nn.Module):
    def __init__(self, feature_size=257*2, latent_size=128, num_embeddings=512, embedding_dim=128):
        super(VQ_VAE, self).__init__()
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # encode
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ELU(),
            nn.Linear(512, latent_size)
        )
        
        # codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # decode
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ELU(),
            nn.Linear(512, feature_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x.view(-1, self.feature_size))  # Get latent representation
        z_quantized, indices = self.vector_quantization(z)  # Quantize the latent space
        x_recon = self.decoder(z_quantized)  # Decode the quantized latent space
        return x_recon, z, z_quantized

    def vector_quantization(self, z):
        distances = torch.cdist(z, self.codebook.weight, p=2.0)  # Compute distances
        indices = torch.argmin(distances, dim=1)  # Find nearest neighbors in the codebook
        z_quantized = self.codebook(indices)  # Quantized latent space
        return z_quantized, indices


if __name__ == '__main__':
    model = VQ_VAE()
    x = torch.randn(10, 257*2)
    x_recon, z, z_quantized = model(x)
    print(x_recon.shape, z.shape, z_quantized.shape)