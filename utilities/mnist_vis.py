import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def exsamples_vis(dataloader,
                  geo_subplots: 'tuple' = (4, 7),
                  figsize: 'tuple' = (7, 6),
                  **kwargs):
    '''Visualize some exsamples of MNIST'''

    logger.info(f"Visualize some exsamples.")

    try:
        mean = kwargs['mean']
        std = kwargs['std']
        unnormal = True
        logger.info(f"Input data will be unnormalized.")
    except KeyError:
        unnormal = False
        logger.info(f"Input data will be normalized.")

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
            img = images[i] * std + mean
        else:
            img = images[i]

        # Reshape the input data
        img = img.reshape([28, 28])
        # Make sure the intensity stays in range of [0, 255]
        img = (img * 255).astype(np.uint8)
        ax[i].imshow(img, cmap="gray")
        ax[i].set_title(f"Label: {labels[i]}")

    plt.show()