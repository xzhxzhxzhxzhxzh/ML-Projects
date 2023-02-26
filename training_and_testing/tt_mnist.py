import torch, logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_preparation.dp_mnist import load_data_mnist
from utilities.mnist_vis import exsamples_vis
from models.model_mnist import Model_MNIST
from training_and_testing.tt_generic import early_stopping

logger = logging.getLogger(__name__)


def mnist(step_size: 'float' = 1,
          batch_size: 'int' = 1000,
          epochs: 'int' = 20,
          lam: 'float' = 0.0,
          data_root: 'str' = './data/mnist',
          model_name: 'list' = ['mnist_1', 'mnist_2'],
          device: 'torch.device' = torch.device("cpu")):
    '''Training and testing on mnist dataset.'''

    tr_loader, val_loader, te_loader, mean, std = load_data_mnist(
        batch_size, data_root)

    # Visualize some exsamples
    exsamples_vis(tr_loader,
                  geo_subplots=(4, 7),
                  figsize=(7, 6),
                  mean=mean,
                  std=std)

    # Training model without regulization
    logger.info(f"Training model without regulization.")
    model_mnist = Model_MNIST(784, 10, 300).to(device)
    optimizer = torch.optim.SGD(model_mnist.parameters(), lr=step_size)
    criterion = torch.nn.CrossEntropyLoss()

    results_1 = training_model(model_mnist,
                               optimizer,
                               criterion,
                               device,
                               model_name=model_name[0],
                               epochs=epochs,
                               training=True,
                               train_loader=tr_loader,
                               valid_loader=val_loader)

    # Training model 2
    logger.info(f"Training model with regulization.")
    model_mnist = Model_MNIST(784, 10, 300).to(device)
    optimizer = torch.optim.SGD(model_mnist.parameters(), lr=step_size)
    criterion = torch.nn.CrossEntropyLoss()

    results_2 = training_model(model_mnist,
                               optimizer,
                               criterion,
                               device,
                               model_name=model_name[1],
                               epochs=epochs,
                               training=True,
                               regulization=True,
                               lam=lam,
                               train_loader=tr_loader,
                               valid_loader=val_loader)

    # Visualize validation results
    logger.info(f"Visualize validation results.")
    _, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(results_1[2], label='Training Accuracy without RE')
    ax[0].plot(results_2[2], label='Training Accuracy with RE')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(results_1[3], label='Validation Accuracy without RE')
    ax[1].plot(results_2[3], label='Validation Accuracy with RE')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].grid()
    ax[1].legend()
    plt.show()

    # Testing model 1
    logger.info(f"Testing model without regulization.")
    model_mnist.load_state_dict(torch.load(f'./outputs/{model_name[0]}.pt'))
    optimizer = torch.optim.SGD(model_mnist.parameters(), lr=step_size)
    criterion = torch.nn.CrossEntropyLoss()

    results_1 = training_model(model_mnist,
                               optimizer,
                               criterion,
                               device,
                               epochs=epochs,
                               training=False,
                               test_loader=te_loader)

    # Testing model 2
    logger.info(f"Testing model with regulization.")
    model_mnist.load_state_dict(torch.load(f'./outputs/{model_name[1]}.pt'))
    optimizer = torch.optim.SGD(model_mnist.parameters(), lr=step_size)
    criterion = torch.nn.CrossEntropyLoss()

    results_2 = training_model(model_mnist,
                               optimizer,
                               criterion,
                               device,
                               epochs=epochs,
                               training=False,
                               regulization=True,
                               lam=lam,
                               test_loader=te_loader)

    # Show test results
    best_acc_1 = round(max(results_1[3]) * 100, 2)
    best_acc_2 = round(max(results_2[3]) * 100, 2)
    logger.info(f"The best accuracy on the test set without regulization:"
                f" {best_acc_1}%")
    logger.info(f"The best accuracy on the test set with regulization:"
                f" {best_acc_2}%")


def training_model(model,
                   optimizer,
                   criterion,
                   device,
                   epochs: 'int',
                   training: 'bool' = True,
                   regulization: 'bool' = False,
                   **kwargs):
    '''Define a training (and testing) process.'''

    if training:
        train_loader = kwargs['train_loader']
        valid_loader = kwargs['valid_loader']
        model_name = kwargs['model_name']
    else:
        train_loader = None
        model_name = None
        valid_loader = kwargs['test_loader']

    if regulization:
        lam = kwargs['lam']
    else:
        lam = 0.0

    # Record loss and accuracy
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []

    for epoch in range(epochs):

        if training:
            logger.info(f"Epoch {epoch+1}: Start training.")

            # Keep track of training loss and training accuracy
            tr_loss, tr_acc = 0.0, 0.0
            model.train()
            for (data, label) in train_loader:
                data, label = torch.squeeze(data).to(device), label.to(device)

                optimizer.zero_grad()
                tr_output = model(data)
                loss = criterion(tr_output, label)

                if regulization:
                    # Add a regularization term
                    w_1 = list(model.parameters())[0]
                    w_2 = list(model.parameters())[1]
                    loss = loss + lam * (torch.linalg.norm(w_1)**2 +
                                         torch.linalg.norm(w_2)**2)

                loss.backward()
                optimizer.step()

                tr_loss += loss.item() * data.size(0)
                tr_acc += mnist_accuracy(tr_output, label, device)

            # Record training loss.
            training_loss.append(tr_loss / len(train_loader.dataset))
            training_acc.append(tr_acc / len(train_loader.dataset))

        with torch.no_grad():
            if training:
                logger.info(f"Epoch {epoch+1}: Start validating.")
            else:
                logger.info(f"Epoch {epoch+1}: Start testing.")

            # Keep track of validation loss and validation accuracy
            val_loss, val_acc = 0.0, 0.0

            # Compute and record validation loss and validation set accuracy
            model.eval()
            for (data, label) in valid_loader:
                data, label = torch.squeeze(data).to(device), label.to(device)
                val_output = model(data)
                loss = criterion(val_output, label)
                val_loss += loss.item() * data.size(0)
                val_acc += mnist_accuracy(val_output, label, device)
                
            validation_loss.append(val_loss / len(valid_loader.dataset))
            validation_acc.append(val_acc / len(valid_loader.dataset))

            if training:
                if early_stopping(validation_loss):
                    torch.save(model.state_dict(),
                               f'./outputs/{model_name}.pt')
                    break
                elif epoch == epochs - 1:
                    torch.save(model.state_dict(),
                               f'./outputs/{model_name}.pt')
                    logger.warning(f"Early stopping is not implemented!")

    return [training_loss, validation_loss, training_acc, validation_acc]


def mnist_accuracy(output, label, device):
    '''Compute accuracy for MNIST.'''

    with torch.no_grad():
        # Compute and record training set accuracy
        pred = F.softmax(output - torch.max(output), dim=1)
        correct = torch.sum(torch.argmax(pred, dim=1) == label)
        correct = correct.detach().cpu().numpy(
        ) if device.type == 'mps' else correct.detach().numpy()
        return correct
