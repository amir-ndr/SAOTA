import torch
from torchvision import datasets, transforms
import numpy as np

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def partition_mnist_noniid(train_dataset, num_clients=10, shards_per_client=2):
    # num_shards = num_clients * shards_per_client
    num_shards = 100
    shard_size = len(train_dataset) // num_shards
    data_indices = np.arange(len(train_dataset))
    labels = np.array(train_dataset.targets)

    sorted_indices = data_indices[np.argsort(labels)]

    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
    np.random.shuffle(shards)

    client_data_map = {i: [] for i in range(num_clients)}
    for i in range(num_clients):
        assigned_shards = shards[i * shards_per_client:(i + 1) * shards_per_client]
        client_data_map[i] = np.concatenate(assigned_shards)

    return client_data_map
def partition_mnist_noniid(train_dataset, num_clients=10, shards_per_client=2):
    num_shards = num_clients * shards_per_client  # Use ALL data
    shard_size = len(train_dataset) // num_shards  # Recompute shard size
    data_indices = np.arange(len(train_dataset))
    labels = np.array(train_dataset.targets)

    sorted_indices = data_indices[np.argsort(labels)]
    shards = [sorted_indices[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)]
    np.random.shuffle(shards)

    client_data_map = {}
    for i in range(num_clients):
        assigned_shards = shards[i * shards_per_client : (i + 1) * shards_per_client]
        client_data_map[i] = np.concatenate(assigned_shards)
    return client_data_map

import numpy as np
import torch

def partition_mnist_dirichlet(train_dataset, num_clients=10, alpha=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    labels = np.array(train_dataset.targets)
    num_classes = len(np.unique(labels))
    client_data_map = {i: [] for i in range(num_clients)}
    idxs = np.arange(len(labels))

    # For each class, split indices among clients using Dirichlet distribution
    for c in range(num_classes):
        class_idxs = idxs[labels == c]
        np.random.shuffle(class_idxs)
        
        # Sample proportions for clients from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([
            p * (len(client_data_map[i]) < len(labels) / num_clients)
            for i, p in enumerate(proportions)
        ])
        proportions /= proportions.sum()
        proportions = (np.cumsum(proportions) * len(class_idxs)).astype(int)[:-1]
        
        # Split class indices and assign to clients
        splits = np.split(class_idxs, proportions)
        for i in range(num_clients):
            client_data_map[i].extend(splits[i].tolist())

    return client_data_map


def partition_mnist_dirichlet2(train_dataset, num_clients=10, alpha=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    labels = np.array(train_dataset.targets)
    num_classes = len(np.unique(labels))
    client_data_map = {i: [] for i in range(num_clients)}
    idxs = np.arange(len(labels))

    for c in range(num_classes):
        class_idxs = idxs[labels == c]
        np.random.shuffle(class_idxs)

        # Sample Dirichlet proportions and normalize
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_idxs)).astype(int)[:-1]
        splits = np.split(class_idxs, proportions)

        for i in range(num_clients):
            client_data_map[i].extend(splits[i].tolist())

    return client_data_map
