import torch, logging
from torchvision import datasets
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


def load_data_cifar10(batch_size: 'int' = 20,
                      data_root: 'str' = './data/cifar10'):
    '''Create training, validation and test dataloader'''

    logger.info(f"Construct transforms.")
    # Specify mean and std of dataset
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Construct transforms
    tr_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    te_transforms = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    logger.info(f"Download the datasets.")
    # Acquire training and test datasets.
    train_dataset = datasets.CIFAR10(root=data_root,
                                    train=True,
                                    download=True,
                                    transform=tr_transforms)
    valid_dataset = datasets.CIFAR10(root=data_root,
                                    train=True,
                                    download=True,
                                    transform=te_transforms)
    test_dataset = datasets.CIFAR10(root=data_root,
                                    train=False,
                                    download=True,
                                    transform=te_transforms)

    logger.info(f"Split training and validation data.")
    # Use 20% of training data for validation.
    train_dataset_size = int(len(train_dataset) * 0.8)
    valid_dataset_size = len(train_dataset) - train_dataset_size

    seed = torch.Generator().manual_seed(42)
    train_set, _ = torch.utils.data.random_split(
        train_dataset, [train_dataset_size, valid_dataset_size], generator=seed)
    _, valid_set = torch.utils.data.random_split(
        valid_dataset, [train_dataset_size, valid_dataset_size], generator=seed)

    logger.info(f"Construct dataloaders.")
    # Construct data loader for training, validation and test sets.
    train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=batch_size,
                                            shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                            batch_size=batch_size,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    return train_loader, valid_loader, test_loader, mean, std
