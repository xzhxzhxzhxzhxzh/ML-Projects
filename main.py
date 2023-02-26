import torch, logging
from utilities.logger_setup import logger_setup
from utilities.device_setup import device_setup
from training_and_testing.tt_cifar10 import cifar10
from training_and_testing.tt_mnist import mnist

if __name__ == "__main__":

    # Initialize logger
    logger_setup()
    logger = logging.getLogger(__name__)

    device = device_setup()

    ############################################################################
    # Specify which datasets to learn
    datasets = ['CIFAR10', 'MNIST']
    #datasets = ['MNIST']
    ############################################################################

    if 'CIFAR10' in datasets:
        logger.info(f"Start analysing CIFAR10.")
        cifar10(step_size=0.01, batch_size=20, epochs=10, device=device)
    elif 'MNIST' in datasets:
        logger.info(f"Start analysing CIFAR10.")
        mnist(step_size=5,
              batch_size=1000,
              epochs=10,
              lam=0.0001,
              device=torch.device("cpu"))
    else:
        logger.error(f"Nothing happens, your computer is survived :)")

    logger.info(f"Program is completed.")