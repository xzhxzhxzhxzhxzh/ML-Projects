import logging

logger = logging.getLogger(__name__)


def early_stopping(val_loss: 'list', error: 'float' = 0.1):
    '''Stop training if validation loss has increased.'''

    if val_loss[-1] - min(val_loss) >= error:
        logger.info(f"Validation loss starts increasing, training will stop.")
        return True
    else:
        return False