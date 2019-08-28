import numpy as np
from autopower.data.datasets import DefaultDataset


def generator_to_array(Generator):

    array_from_generator = np.array(list(Generator)).T

    labels, data = array_from_generator

    return np.asarray(labels), np.asarray(data)


def get_train_validation_split(hdf_file_path, train_size, validation_size):

    data_generator_train = DefaultDataset(
            as_tensor = False, mode = 'training',
            hdf_file_path = hdf_file_path,
            train_size = train_size,
            validation_size = validation_size
            )

    train_labels, train_data = generator_to_array(data_generator_train)

    data_generator_val = DefaultDataset(
            as_tensor = False, mode = 'validation',
            hdf_file_path = hdf_file_path,
            train_size = train_size,
            validation_size = validation_size
            )


    val_labels, val_data = generator_to_array(data_generator_val)


    return np.stack(train_data, axis = 0), np.stack(train_labels, axis = 0), \
            np.stack(val_data, axis = 0), np.stack(val_labels, axis = 0)


