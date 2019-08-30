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
                 n_hidden_layers: int = 2,
                 n_hidden_size: int = 512,
                 n_latent_dim: int = 2):

        super(DefaultAutoencoder, self).__init__()
        
        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        
        self.n_input_dim = n_input_dim
        self.n_latent_dim = n_latent_dim
        
        # ---------------------------------------------------------------------
        # Define the encoder's layers
        # ---------------------------------------------------------------------

        '''
        if n_hidden_layers == 1:
            n_hidden_size = n_latent_dim


        layers  = [ nn.Linear(in_features=n_input_dim, out_features = n_hidden_size),
                    nn.ReLU()]

        for hidden in range(n_hidden_layers):
            layers.append( 
                    nn.Linear( in_features = n_hidden_size,
                                out_features = n_hidden_size
                            ) 
                    )
            layers.append(nn.ReLU())

        if n_hidden_layers > 1:
            layers.append(
                    nn.Linear(
                        in_features = n_hidden_size,
                        out_features = n_latent_dim
                        )
                    )
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        '''

        self.encoder = nn.Sequential( nn.Linear( in_features = n_input_dim,
                                    out_features = n_hidden_size), nn.ReLU(),
                                    nn.Linear( in_features = n_hidden_size, out_features = n_latent_dim))


        # ---------------------------------------------------------------------
        # Define the decoder's layers
        # ---------------------------------------------------------------------

        '''
        if n_hidden_layers == 1:
            n_hidden_size = n_latent_dim


        if n_hidden_layers > 1:
            layers  = [ nn.Linear(in_features=n_latent_dim,
                            out_features = n_hidden_size),
                        nn.ReLU()]

        else:
            layers = [nn.Linear(in_features = n_latent_dim,
                out_features = n_input_dim)]

        for hidden in range(n_hidden_layers):
            layers.append( 
                    nn.Linear( in_features = n_hidden_size,
                                out_features = n_hidden_size
                            )
                    )
            layers.append(nn.ReLU())

        if n_hidden_layers > 1:
            layers.append(
                    nn.Linear(
                        in_features = n_hidden_size,
                        out_features =n_input_dim 
                        )
                    )

        self.decoder= nn.Sequential(*layers)
        '''

        self.decoder = nn.Sequential( nn.Linear( in_features = n_latent_dim, out_features = n_hidden_size),
                                nn.ReLU(), nn.Linear(in_features = n_hidden_size, out_features = n_input_dim))



        

    # -------------------------------------------------------------------------

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ConvAutoencoder(nn.Module):

    def __init__(self,
                 n_input_dim: int = 200,
                 n_hidden_layers: int = 3,
                 n_latent_dim: int = 2):

        super(ConvAutoencoder, self).__init__()
        
        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        
        self.n_input_dim = n_input_dim
        self.n_latent_dim = n_latent_dim
        
        # ---------------------------------------------------------------------
        # Define the encoder's layers
        # ---------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv1d(6, 16,kernel_size=5),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(in_features = 16 * ((200 - 5 + 1) - 5 + 1), out_features = n_latent_dim)
            )        


        # ---------------------------------------------------------------------
        # Define the decoder's layers
        # ---------------------------------------------------------------------

        self.decoder = nn.Sequential(             
            nn.ConvTranspose1d(1,16,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose1d(16,1,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose1d(6,1,kernel_size=5),
            )

    # -------------------------------------------------------------------------

    def forward(self, x):

        x = x[:, None, :]
        x = self.encoder(x)
        #x = self.decoder(x)

        return x



# -----------------------------------------------------------------------------
# MAIN CODE (= BASIC TESTING ZONE)
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Instantiate the default model
    print('Instantiating model...', end=' ', flush=True)
    model = ConvAutoencoder()
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
