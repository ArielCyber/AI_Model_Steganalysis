import os, sys
import torch
import numpy as np
from stego import float_to_binary

from embedding import embed_s, binzoo_to_floatzoo


def request_logger(logger_name: str = None):
    if logger_name is None:
        logger_name = __name__

    import logging

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger

logger = request_logger()

def log_it(func):
    def wrap(*args, **kwargs):
        func_name = func.__name__
        
        args_str = ', '.join([str(arg) for arg in args])
        kwargs_str = ' | '.join(map(lambda tup: f'{tup[0]}:{tup[1]}', sorted(kwargs.items())))

        run_str = f'running {func_name} with args: [{args_str}] | kwargs: {kwargs_str}'
        logger.info(run_str)

        try:
            ret = func(*args, **kwargs)
        except:
            logger.exception(f'~~failed~~ {run_str}')
            return None

        logger.info(f'~~finished~~ {run_str}')
        return ret
    
    return wrap

def get_zoo_path(zoo_name='mnist'):
    return os.path.join('data', 'model_zoos', zoo_name)

def get_zoo_weights_path(zoo_name='mnist'):
    zoo_dir_path = get_zoo_path(zoo_name)
    return os.path.join(zoo_dir_path, 'zoo_weights.npy')

def get_zoo_weights_bin_path(zoo_name='mnist'):
    zoo_dir_path = get_zoo_path(zoo_name)
    return os.path.join(zoo_dir_path, 'zoo_weights_bin.npz')

def get_embedded_weights_path(zoo_name='mnist', malware_name='malware_2448b', fill=True, msb=False):
    return os.path.join('data', 'embedded', f'zoo:{zoo_name}_malwarename:{malware_name}_fill:{fill}_msb:{msb}.npz')

def get_grads_path(zoo_name='mnist', malware_name='malware_2448b', fill=True, msb=False):
    return os.path.join('data', 'features', 'grads', f'zoo:{zoo_name}_malwarename:{malware_name}_fill:{fill}_msb:{msb}.npz')

def get_losses_path(zoo_name='mnist', malware_name='malware_2448b', fill=True, msb=False):
    return os.path.join('data', 'features', 'losses', f'zoo:{zoo_name}_malwarename:{malware_name}_fill:{fill}_msb:{msb}.npz')

@log_it
def create_zoo_weights(zoo_name='mnist'):
    # log_str = f'creating zoo weights for {zoo_name}'
    # logger.info(log_str)

    zoo_dir_path = get_zoo_path(zoo_name)
    save_path = get_zoo_weights_path(zoo_name)
    dataset_path = os.path.join(zoo_dir_path, 'dataset.pt')

    dataset = torch.load(dataset_path)

    trainset = dataset['trainset'].__get_weights__()
    testset = dataset['testset'].__get_weights__()
    valset = dataset.get('valset', None).__get_weights__()

    all_weights = torch.cat((trainset, testset, valset), 0)
    all_weights = all_weights.numpy()

    np.save(save_path, all_weights)

@log_it
def create_zoo_weights_bin(zoo_name='mnist'):
    # logger.info(f'creating binary zoo weights for {zoo_name}')

    save_path = get_zoo_weights_bin_path(zoo_name)
    zoo_weights = ret_zoo_weights(zoo_name)
    zoo_weights_bin = np.array([[float_to_binary(v) for v in arr] for arr in zoo_weights])

    np.savez_compressed(save_path, bin=zoo_weights_bin)

@log_it
def create_embedded_weights(zoo_name='mnist', malware_name='malware_12584bytes', msb=False, fill=True):
    # logger.info(f'creating embedded weights for {zoo_name} | malware_name:{malware_name} | msb:{msb} | fill:{fill}')
    # exports = {}

    # zoo_weights = ret_zoo_weights(zoo_name)

    # for lsb in range(23,0,-1):
    #     embedded_weights = modify(zoo_weights, malware_name, lsb=lsb, msb=msb, fill=fill)
    #     exports[str(lsb)] = embedded_weights

    # save_path = get_embedded_weights_path(zoo_name=zoo_name, malware_name=malware_name, fill=fill, msb=msb)
    # np.savez_compressed(save_path, **exports)
    zoo_weights_bin = ret_zoo_weights_bin(zoo_name)
    with open(os.path.join('data', 'malware', 'malware_12584bytes'), 'rb') as malware_file:
        malware_str = malware_file.read()

    exports = {}
    for lsb in range(23,0,-1):
        host_embedded_bin = embed_s(zoo_weights_bin, malware_str, lsb=lsb, fill=fill, inplace=False, msb=msb)

        host_embedded = binzoo_to_floatzoo(host_embedded_bin)
        exports[str(lsb)] = host_embedded

    save_path = get_embedded_weights_path(zoo_name=zoo_name, malware_name=malware_name, fill=fill, msb=msb)
    np.savez_compressed(save_path, **exports)


def ret_zoo_weights(zoo_name='mnist'):
    zoo_weights_path = get_zoo_weights_path(zoo_name)

    if not os.path.isfile(zoo_weights_path):
        logger.info(f'zoo weights for {zoo_name} not found')
        create_zoo_weights(zoo_name)

    zoo_weights = np.load(zoo_weights_path)
    return zoo_weights

def ret_zoo_weights_bin(zoo_name='mnist'):
    zoo_weights_bin_path = get_zoo_weights_bin_path(zoo_name)

    if not os.path.isfile(zoo_weights_bin_path):
        logger.info(f'binary zoo weights for {zoo_name} not found')
        create_zoo_weights_bin(zoo_name)

    with np.load(zoo_weights_bin_path) as data:
        zoo_weights_bin = data['bin']

    return zoo_weights_bin

def ret_embedded_weights(zoo_name='mnist', malware_name='malware_12584bytes', lsb=-1, msb=False, fill=True):
    embedded_weights_path = get_embedded_weights_path(zoo_name=zoo_name, malware_name=malware_name, fill=fill, msb=msb)

    if not os.path.isfile(embedded_weights_path):
        create_embedded_weights(zoo_name=zoo_name, malware_name=malware_name, msb=msb, fill=fill)

    embedded_weights = np.load(embedded_weights_path)
    if lsb==-1:
        return embedded_weights
    else:
        requested_embedded_weights=embedded_weights[str(lsb)]
        embedded_weights.close()
        return requested_embedded_weights
    
def ret_grads(zoo_name='mnist', malware_name='malware_12584bytes', lsb=-1, msb=False, fill=True):
    assert lsb==-1 or lsb in range(1,24), 'lsb must be -1 or in range(1,24)'
    
    grads_path = get_grads_path(zoo_name=zoo_name, malware_name=malware_name, fill=fill, msb=msb)

    if not os.path.isfile(grads_path):
        return None
    
    grads = np.load(grads_path)
    if lsb==-1:
        return grads
    else:
        unmodified_grads=grads[str(0)]
        modified_grads=grads[str(lsb)]
        grads.close()
        return unmodified_grads, modified_grads

def ret_losses(zoo_name='mnist', malware_name='malware_12584bytes', fill=True, msb=False):
    losses_path = get_losses_path(zoo_name=zoo_name, malware_name=malware_name, fill=fill, msb=msb)

    if not os.path.isfile(losses_path):
        return None
    
    losses = np.load(losses_path)
    return losses



