import torch, logging

logger = logging.getLogger(__name__)


def device_setup():
    '''Check if MPS is available.'''

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.warning(
                "The current PyTorch install was not built with MPS enabled.")
        else:
            logger.warning(
                "The current MacOS version is not 12.3+ and/or you do not have "
                "an MPS-enabled device on this machine.")
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
        logger.info("MPS is available now.")
    
    return device