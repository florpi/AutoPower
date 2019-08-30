"""
Define fully-connected models.
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

class DefaultFullyConnectedModel(nn.Module):
    
    def __init__(self,
                 n_input_dim: int = 2,
                 n_hidden_layers: int = 3,
                 n_latent_dim: int = 512,
                 n_output_dim: int = 200):
        super(DefaultFullyConnectedModel, self).__init__()
        
        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        
        self.n_input_dim = n_input_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_latent_dim = n_latent_dim
        self.n_output_dim = n_output_dim

        # ---------------------------------------------------------------------
        # Define the model's layers
        # ---------------------------------------------------------------------

        # Define the input layer: n_input_dim -> n_latent_dim
        self.input_layer = nn.Linear(in_features=n_input_dim,
                                     out_features=n_latent_dim)

        # Define a stack of hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            layer = nn.Linear(in_features=n_latent_dim,
                              out_features=n_latent_dim)
            self.hidden_layers.append(layer)

        # Define the output layer: n_latent_dim -> n_input_dim
        self.output_layer = nn.Linear(in_features=n_latent_dim,
                                      out_features=n_output_dim)

    # -------------------------------------------------------------------------

    def forward(self, x):

        x = self.input_layer.forward(x)
        x = torch.relu(x)

        for layer in self.hidden_layers:
            x = layer.forward(x)
            x = torch.relu(x)

        x = self.output_layer.forward(x)

        #return torch.relu(x)
        return x


# -----------------------------------------------------------------------------
# MAIN CODE (= BASIC TESTING ZONE)
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Instantiate the default model
    print('Instantiating model...', end=' ', flush=True)
    model = DefaultFullyConnectedModel()
    print('Done!', flush=True)
    
    # Compute the number of trainable parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([np.prod(p.size()) for p in parameters])
    print('Number of trainable parameters:', n_parameters, '\n')
    
    # Create some dummy input
    print('Creating random input...', end=' ', flush=True)
    data = torch.randn((3, 2))
    print('Done!', flush=True)
    print('Input shape:', data.shape, '\n')
    
    # Compute the forward pass through the model
    print('Computing forward pass...', end=' ', flush=True)
    output = model.forward(data)
    print('Done!', flush=True)
    print('Output shape:', output.shape)
