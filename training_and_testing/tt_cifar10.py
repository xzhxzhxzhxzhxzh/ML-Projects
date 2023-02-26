import torch, logging
import matplotlib.pyplot as plt
from data_preparation.dp_cifar10 import load_data_cifar10
from utilities.cifar10_vis import exsamples_vis
from models.model_1 import Model_1
from models.model_2 import Model_2
from models.model_3 import Model_3
from training_and_testing.tt_generic import early_stopping

logger = logging.getLogger(__name__)


def cifar10(step_size: 'float' = 0.01,
            batch_size: 'int' = 20,
            epochs: 'int' = 20,
            data_root: 'str' = './data/cifar10',
            model_name: 'list' = ['model_1', 'model_2', 'model_3'],
            device: 'torch.device' = torch.device("cpu")):
    '''Training and testing on cifar10 dataset.'''

    tr_loader, val_loader, te_loader, mean, std = load_data_cifar10(
        batch_size, data_root)

    # Specify the image classes
    label_mapping = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    exsamples_vis(tr_loader,
                  geo_subplots=(3, 6),
                  figsize=(7, 5),
                  mean=mean,
                  std=std,
                  label_mapping=label_mapping)

    # Training model 1
    logger.info(f"Training model 1.")
    model_1 = Model_1().to(device)
    optimizer = torch.optim.SGD(model_1.parameters(), lr=step_size)
    criterion = torch.nn.CrossEntropyLoss()

    results_1 = training_model(model_1,
                               optimizer,
                               criterion,
                               device,
                               model_name=model_name[0],
                               epochs=epochs,
                               training=True,
                               train_loader=tr_loader,
                               valid_loader=val_loader)

    # Training model 2
    logger.info(f"Training model 2.")
    model_2 = Model_2().to(device)
    optimizer = torch.optim.SGD(model_2.parameters(), lr=step_size)
    criterion = torch.nn.CrossEntropyLoss()

    results_2 = training_model(model_2,
                               optimizer,
                               criterion,
                               device,
                               model_name=model_name[1],
                               epochs=epochs,
                               training=True,
                               train_loader=tr_loader,
                               valid_loader=val_loader)

    # Training model 3
    logger.info(f"Training model 3.")
    model_3 = Model_3().to(device)
    optimizer = torch.optim.SGD(model_3.parameters(), lr=step_size)
    criterion = torch.nn.CrossEntropyLoss()

    results_3 = training_model(model_3,
                               optimizer,
                               criterion,
                               device,
                               model_name=model_name[2],
                               epochs=epochs,
                               training=True,
                               train_loader=tr_loader,
                               valid_loader=val_loader)

    # Visualize validation results
    logger.info(f"Visualize validation results.")
    _, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(results_1[2], label='Training Accuracy of Model 1')
    ax[0].plot(results_2[2], label='Training Accuracy of Model 2')
    ax[0].plot(results_3[2], label='Training Accuracy of Model 3')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(results_1[3], label='Validation Accuracy of Model 1')
    ax[1].plot(results_2[3], label='Validation Accuracy of Model 2')
    ax[1].plot(results_3[3], label='Validation Accuracy of Model 3')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].grid()
    ax[1].legend()
    plt.show()

    # Testing model 1
    logger.info(f"Testing model 1.")
    model_1.load_state_dict(torch.load(f'./outputs/{model_name[0]}.pt'))
    optimizer = torch.optim.SGD(model_1.parameters(), lr=step_size)
    criterion = torch.nn.CrossEntropyLoss()

    results_1 = training_model(model_1,
                               optimizer,
                               criterion,
                               device,
                               epochs=epochs,
                               training=False,
                               test_loader=te_loader)

    # Testing model 2
    logger.info(f"Testing model 2.")
    model_2.load_state_dict(torch.load(f'./outputs/{model_name[1]}.pt'))
    optimizer = torch.optim.SGD(model_2.parameters(), lr=step_size)
    criterion = torch.nn.CrossEntropyLoss()

    results_2 = training_model(model_2,
                               optimizer,
                               criterion,
                               device,
                               epochs=epochs,
                               training=False,
                               test_loader=te_loader)

    # Testing model 3
    logger.info(f"Testing model 3.")
    model_3.load_state_dict(torch.load(f'./outputs/{model_name[2]}.pt'))
    optimizer = torch.optim.SGD(model_3.parameters(), lr=step_size)
    criterion = torch.nn.CrossEntropyLoss()

    results_3 = training_model(model_3,
                               optimizer,
                               criterion,
                               device,
                               epochs=epochs,
                               training=False,
                               test_loader=te_loader)

    # Show test results
    best_acc_1 = round(max(results_1[3]) * 100, 2)
    best_acc_2 = round(max(results_2[3]) * 100, 2)
    best_acc_3 = round(max(results_3[3]) * 100, 2)
    logger.info(
        f"The best accuracy on the test set using model 1: {best_acc_1}%")
    logger.info(
        f"The best accuracy on the test set using model 2: {best_acc_2}%")
    logger.info(
        f"The best accuracy on the test set using model 3: {best_acc_3}%")


def training_model(model,
                   optimizer,
                   criterion,
                   device,
                   epochs: 'int',
                   training: 'bool' = True,
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
                data, label = data.to(device), label.to(device)

                optimizer.zero_grad()
                tr_output = model(data)
                loss = criterion(tr_output, label)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item() * data.size(0)
                tr_acc += cifar10_accuracy(tr_output, label,
                                           device) * data.size(0)

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
                data, label = data.to(device), label.to(device)
                val_output = model(data)
                loss = criterion(val_output, label)
                val_loss += loss.item() * data.size(0)
                val_acc += cifar10_accuracy(val_output, label,
                                            device) * data.size(0)

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


def cifar10_accuracy(output, label, device):
    '''Compute accuracy for CIFAR10.'''

    _, pred = torch.max(output, dim=1)
    pred = pred.detach().cpu().numpy(
    ) if device.type == 'mps' else pred.detach().numpy()
    label = label.detach().cpu().numpy(
    ) if device.type == 'mps' else label.detach().numpy()
    accuracy = sum(pred == label) / len(pred)
    return accuracy
