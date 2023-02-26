import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def exsamples_vis(dataloader,
                  geo_subplots: 'tuple' = (2, 5),
                  figsize: 'tuple' = (3, 5),
                  **kwargs):
    '''Visualize some exsamples of CIFAR10'''

    logger.info(f"Visualize some exsamples.")

    try:
        mean = kwargs['mean']
        std = kwargs['std']
        unnormal = True
        logger.info(f"Input data will be unnormalized.")
    except KeyError:
        unnormal = False
        logger.info(f"Input data will be normalized.")

    try:
        label_mapping = kwargs['label_mapping']
        mapping = True
    except KeyError:
        mapping = False

    exsamples = iter(dataloader)
    images, labels = next(exsamples)
    images = images.numpy()
    labels = labels.numpy()

    _, ax = plt.subplots(geo_subplots[0],
                         geo_subplots[1],
                         sharey=True,
                         figsize=figsize)
    ax = ax.ravel()
    for i in range(np.prod(geo_subplots)):
        if unnormal:
            # Unnormalize
            img = images[i] * np.array(std)[:, None, None] + np.array(
                mean)[:, None, None]
        else:
            img = images[i]

        # Make sure the color intensity stays in range of [0, 255]
        img = (img * 255).astype(np.uint8)
        ax[i].imshow(np.transpose(img, (1, 2, 0)))
        if mapping:
            ax[i].set_title(f"{label_mapping[labels[i]]}")
        else:
            ax[i].set_title(f"{labels[i]}")

    plt.show()