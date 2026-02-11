import torch
from torchvision import datasets, transforms
import numpy as np

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def partition_mnist_dirichlet(train_dataset, num_clients=10, alpha=0.5, seed=None, min_size=10):
    rng = np.random.default_rng(seed)
    labels = np.array(train_dataset.targets)
    num_classes = len(np.unique(labels))
    idxs = np.arange(len(labels))

    # Add safeguard to prevent infinite loops with extreme alpha values
    max_iterations = 100
    iteration = 0

    while iteration < max_iterations:
        client_data_map = {i: [] for i in range(num_clients)}

        for c in range(num_classes):
            class_indices = idxs[labels == c]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            splits = np.split(class_indices, split_points)

            for i in range(num_clients):
                client_data_map[i].extend(splits[i].tolist())

        sizes = np.array([len(client_data_map[i]) for i in range(num_clients)])
        if sizes.min() >= min_size:
            return client_data_map
        
        iteration += 1

    # If max iterations exceeded, return best attempt and warn
    print(f"⚠️  Max iterations ({max_iterations}) reached for MNIST partition.")
    print(f"   Min client size: {sizes.min()}, requested: {min_size}")
    print(f"   Returning partition anyway. Consider increasing min_size or alpha.")
    return client_data_map

def load_cifar10(root="./data", download=True):
    # Common CIFAR-10 normalization
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, download=download, transform=transform_train)
    test_dataset  = datasets.CIFAR10(root=root, train=False, download=download, transform=transform_test)
    return train_dataset, test_dataset

def partition_cifar10_dirichlet(train_dataset, num_clients=10, alpha=0.5, seed=None, min_size=10):
    """
    Dirichlet non-IID partition for CIFAR-10.

    Returns:
        client_data_map: dict {client_id: [sample_indices]}
    """
    rng = np.random.default_rng(seed)

    # CIFAR-10 labels are in train_dataset.targets (list of ints)
    labels = np.array(train_dataset.targets)
    num_classes = len(np.unique(labels))
    idxs = np.arange(len(labels))

    # Add safeguard to prevent infinite loops with extreme alpha values
    max_iterations = 100
    iteration = 0

    while iteration < max_iterations:
        client_data_map = {i: [] for i in range(num_clients)}

        for c in range(num_classes):
            class_indices = idxs[labels == c]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            splits = np.split(class_indices, split_points)

            for i in range(num_clients):
                client_data_map[i].extend(splits[i].tolist())

        sizes = np.array([len(client_data_map[i]) for i in range(num_clients)])
        if sizes.min() >= min_size:
            return client_data_map
        
        iteration += 1

    # If max iterations exceeded, return best attempt and warn
    print(f"⚠️  Max iterations ({max_iterations}) reached for CIFAR-10 partition.")
    print(f"   Min client size: {sizes.min()}, requested: {min_size}")
    print(f"   Returning partition anyway. Consider increasing min_size or alpha.")
    return client_data_map
