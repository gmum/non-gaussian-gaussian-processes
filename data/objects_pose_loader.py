# import numpy as np
import torch
# import torchvision.transforms as transforms
# from PIL import Image
import pickle
import numpy as np
import os


def get_objects_batch(x, y, meta_batch_size, update_batch_size, num_classes):
    inputs = []
    targets = []

    for _ in range(meta_batch_size):
        # sample WAY classes
        classes = np.random.choice(range(np.shape(x)[0]), size=num_classes, replace=False)

        x_list = []
        y_list = []
        for k in list(classes):
            # sample SHOT and QUERY instances
            idx = np.random.choice(range(np.shape(x)[1]), size=update_batch_size + update_batch_size, replace=False)
            x_k = x[k][idx]
            y_k = y[k][idx]

            x_list.append(x_k)
            y_list.append(y_k)

        x_list = np.concatenate(x_list, 0)
        y_list = np.concatenate(y_list, 0)

        inputs.append(x_list)
        targets.append(y_list)

    inputs = np.stack(inputs, 0)
    targets = np.stack(targets, 0)

    inputs = np.reshape(inputs, [meta_batch_size, (update_batch_size + update_batch_size) * num_classes, -1])
    targets = np.reshape(targets, [targets.shape[0], targets.shape[1]])

    inputs = inputs.astype(np.float32) / 255.0
    targets = targets.astype(np.float32) * 10.0

    return torch.from_numpy(inputs), torch.from_numpy(targets)


def get_dataset(train=True, prefix="./filelists/objects_pose"):
    if train:
        file_path = os.path.join(prefix,'train_data.pkl')
    else:
        file_path = os.path.join(prefix,'val_data.pkl')
    with open(file_path, 'rb') as file:
        x, y = pickle.load(file)
    x = np.array(x)
    y = np.array(y)
    y = y[:, :, -1, None]
    return x, y

