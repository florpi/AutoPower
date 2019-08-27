"""
Define autoencoder models (non-variational).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# MODEL DEFINITIONS
# -----------------------------------------------------------------------------

class DefaultAutoencoder(nn.Module):

    def __init__(self,
                 n_input_dim: int = 200,
                 n_latent_dim: int = 6):

        super(DefaultAutoencoder, self).__init__()
        
        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        
        self.n_input_dim = n_input_dim
        self.n_latent_dim = n_latent_dim
        
        # ---------------------------------------------------------------------
        # Define the model's layers
        # ---------------------------------------------------------------------
        
        # Define the input layer: n_input_dim -> n_latent_dim
        self.input_layer = nn.Linear(in_features=n_input_dim,
                                     out_features=n_latent_dim)
        
        # Define the output layer: n_latent_dim -> n_input_dim
        self.output_layer = nn.Linear(in_features=n_latent_dim,
                                      out_features=n_input_dim)

    # -------------------------------------------------------------------------

    def forward(self, x):

        x = self.input_layer.forward(x)
        x = torch.tanh(x)
        x = self.output_layer.forward(x)

        return x


# -----------------------------------------------------------------------------
# MAIN CODE (= BASIC TESTING ZONE)
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Instantiate the default model
    print('Instantiating model...', end=' ', flush=True)
    model = DefaultAutoencoder()
    print('Done!', flush=True)

    # Compute the number of trainable parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([np.prod(p.size()) for p in parameters])
    print('Number of trainable parameters:', n_parameters, '\n')

    # Create some dummy input
    print('Creating random input...', end=' ', flush=True)
    data = torch.randn((3, 200))
    print('Done!', flush=True)
    print('Input shape:', data.shape, '\n')

    # Compute the forward pass through the model
    print('Computing forward pass...', end=' ', flush=True)
    output = model.forward(data)
    print('Done!', flush=True)
    print('Output shape:', output.shape)
