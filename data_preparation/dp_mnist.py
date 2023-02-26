import torch, logging
from torchvision import datasets
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class Reshape(object):
    '''A custom transform to reshape input data'''

    def __init__(self):
        pass

    def __call__(self, img):
        return img.view(1, -1)


def load_data_mnist(batch_size: 'int' = 1000,
                    data_root: 'str' = './data/mnist'):
    '''Create training, validation and test dataloader'''

    logger.info(f"Construct transforms.")
    # Specify mean and std of dataset
    mean = (0.1307, )
    std = (0.3081, )

    # Construct transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std),
         Reshape()])

    logger.info(f"Download the datasets.")
    # Acquire training and test datasets.
    train_dataset = datasets.MNIST(root=data_root,
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root=data_root,
                                  train=False,
                                  download=True,
                                  transform=transform)

    logger.info(f"Split training and validation data.")
    # Use 20% of training data for validation.
    train_dataset_size = int(len(train_dataset) * 0.8)
    valid_dataset_size = len(train_dataset) - train_dataset_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = torch.utils.data.random_split(
        train_dataset, [train_dataset_size, valid_dataset_size],
        generator=seed)

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
