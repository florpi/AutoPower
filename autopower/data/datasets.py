"""
Provide dataset classes that encapsulate access to the data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import h5py
import numpy as np
import torch
import torch.utils.data


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class DefaultDataset(torch.utils.data.Dataset):

    def __init__(self,
                 mode: str,
                 hdf_file_path: str,
                 train_size: int = 1000,
                 validation_size: int = 1000,
                 random_seed: int = 42,
                 as_tensor: bool = True):

        self.as_tensor = as_tensor

        # Basic sanity check: Must select valid mode
        if mode not in ("training", "validation"):
            raise ValueError('mode must be either "training" or "validation"!')

        # Load data from HDF sample file
        with h5py.File(hdf_file_path, 'r') as hdf_file:
            self.data = np.array(hdf_file['pk_ratio'])
            self.labels = np.column_stack((np.array(hdf_file['h']),
                                           np.array(hdf_file['omc'])))

        # Make sure we don't ask for too much data
        if train_size + validation_size > len(self.data):
            raise ValueError("train_size + validation_size is larger than the"
                             "total number of samples in the data set!")

        # Shuffle the data in a reproducible way
        rng = np.random.RandomState(seed=random_seed)
        idx = rng.permutation(list(range(len(self.data))))
        self.data = self.data[idx]
        self.labels = self.labels[idx]

        # Select the right subset based on the mode
        if mode == 'training':
            self.data = self.data[:train_size]
            self.labels = self.labels[:train_size]
        else:
            self.data = self.data[-validation_size:]
            self.labels = self.labels[-validation_size:]

    # -------------------------------------------------------------------------

    def __len__(self):

        return len(self.data)

    # -------------------------------------------------------------------------

    def __getitem__(self, index):
    
        # If desired, convert data to torch tensor first
        if self.as_tensor:
            data = torch.tensor(self.data[index]).float()
            label = torch.tensor(self.labels[index]).float()
        else:
            data = self.data[index]
            label = self.labels[index]

        return data, label
