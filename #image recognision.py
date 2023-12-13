import pickle
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def unpickle(file_path):
    with open(file_path, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data


def cifar_10_reshape(data_batch_arg):
    # Assuming data_batch_arg has a compatible size for reshaping
    # Reshape based on the actual size
    output = np.reshape(data_batch_arg, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return output


def display_images(images, labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i]])
    plt.show()


# Add data augmentation transformations
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Load CIFAR-10 data with the new transformations
train_dataset = CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

batch_size = 64  # Adjust this according to your needs
num_workers = 2  # Adjust this according to your system

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

# Initialize empty lists to store data and labels
data_list = []
labels_list = []

# Iterate over the train_loader to access batches
for batch in train_loader:
    data_batch, labels_batch = batch
    data_batch_reshaped = cifar_10_reshape(data_batch.numpy())
    data_list.append(data_batch_reshaped)
    labels_list.append(labels_batch.numpy())

# Concatenate data and labels
data = np.vstack(data_list)
labels = np.hstack(labels_list)

# Display sample images
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
display_images(data, labels, class_names)
