import torch
import json
from pathlib import Path
import numpy as np

from ghrp.model_definitions.def_net import NNmodule
from ghrp.model_definitions.def_simclr_ae_module import SimCLRAEModule
from ghrp.model_definitions.components.def_loss import LayerWiseReconLoss

import os, sys
from joblib import Parallel, delayed

from data_locator import get_zoo_path, get_grads_path, get_losses_path, ret_zoo_weights, ret_embedded_weights
from data_locator import request_logger, log_it

logger = request_logger(__name__)

@log_it
def create_grads_dataset(zoo_name='mnist', all_layers = False, malware_name = 'malware_12584bytes', msb = False, n_jobs = -1):
    def create_grads(zoo_weights):
        def load_weights_from_flattened_vector(model_weights): 
            state_dict = model_curr.state_dict()
            torch.nn.utils.vector_to_parameters(model_weights, state_dict.values())

            model_curr.load_state_dict(state_dict)

        if n_jobs != 1:
            model_curr = NNmodule(config_model)
            model_curr.to(device)

            layers_curr = [model_curr.model.module_list[-1]]
        else:
            model_curr = model
            layers_curr = layers

        grads = [None]*len(zoo_weights)

        for i, model_weights in enumerate(zoo_weights):
            load_weights_from_flattened_vector(model_weights)

            model_curr.zero_grad()
            outputs = model_curr(inputs)
            
            loss = model_curr.criterion(outputs, labels)
            loss.backward()

            tensors = [None]*len(layers_curr)
            for il, layer in enumerate(layers_curr):
                weights_flattened = torch.flatten(layer.weight.grad)
                bias_flattened = torch.flatten(layer.bias.grad)

                tensors[il] = torch.cat((weights_flattened, bias_flattened))
            
            grads[i] = torch.cat(tensors)

        grads = torch.vstack(tuple(grads))
        return grads
    
    def create_grads_and_export(j):
        modified_weights = torch.from_numpy(embedded_weights[str(j)])
        grads = create_grads(modified_weights)
        exports[str(j)] = grads.cpu().numpy()
        logger.info(f'Finished creating grads: lsb:{j}')

    # logger.info(f'Creating grads dataset for {zoo_name} | all_layers: {all_layers} | malware_name: {malware_name} | msb: {msb} | n_jobs: {n_jobs}')

    """
    All the model zoo code below is from https://github.com/HSG-AIML/NeurIPS_2022-Generative_Hyper_Representations.git
    """

    zoo_dir = get_zoo_path(zoo_name)
    PATH_ROOT = Path(zoo_dir)
    gpus = 1 if torch.cuda.is_available() else 0
    device = torch.device('cuda') if gpus>0 else torch.device('cpu')

    config_model_path = PATH_ROOT.joinpath('config_zoo.json')
    config_model = json.load(config_model_path.open('r'))
    model = NNmodule(config_model)
    model.to(device)

    if zoo_name == 'mnist':
        inputs = torch.zeros((1,1,28,28))
    elif zoo_name == 'cifar10':
        inputs = torch.zeros((1,3,28,28))
    elif zoo_name == 'stl10':
        inputs = torch.zeros((1,3,28,28))
    elif zoo_name == 'svhn':
        inputs = torch.zeros((1,1,28,28))
    labels = torch.zeros((1), dtype=int)

    inputs = inputs.to(device)
    labels = labels.to(device)
    layers = [model.model.module_list[-1]]
    
    zoo_weights = torch.from_numpy(ret_zoo_weights(zoo_name))

    exports = {}
    grads = create_grads(zoo_weights)
    exports['0'] = grads.cpu().numpy()
    logger.info(f'Finished STEP 1: calculating grads from unmodified zoo')

    with ret_embedded_weights(zoo_name=zoo_name, malware_name=malware_name, lsb=-1, msb=msb, fill=True) as embedded_weights:
        Parallel(n_jobs=n_jobs, prefer="threads")(delayed(create_grads_and_export)(j) for j in range(23,0,-1))

    logger.info(f'Finished STEP 2: calculating grads from modified zoo')
    save_path = get_grads_path(zoo_name=zoo_name, malware_name=malware_name, fill=True, msb=msb)

    np.savez_compressed(save_path, **exports)

@log_it
def create_losses_dataset(zoo_name='mnist', malware_name = 'malware_12584bytes', msb = False):
    """
    All the AE code below is from https://github.com/HSG-AIML/NeurIPS_2022-Generative_Hyper_Representations.git
    It is used to load the AE model and calculate the losses
    """
    zoo_dir = get_zoo_path(zoo_name)
    PATH_ROOT = Path(zoo_dir)

    config_path = PATH_ROOT.joinpath('config_ae.json')
    config = json.load(config_path.open('r'))
    config['dataset::dump'] = os.path.abspath(PATH_ROOT.joinpath('dataset.pt'))

    gpus = 1 if torch.cuda.is_available() else 0
    device = torch.device('cuda') if gpus>0 else torch.device('cpu')
    config['device'] = device
    config['model::type'] = 'transformer'

    module = SimCLRAEModule(config)

    checkpoint_path = PATH_ROOT.joinpath('checkpoint_ae.pt')
    checkpoint = torch.load(checkpoint_path,map_location=device)

    module.model.load_state_dict(checkpoint)

    # load dataset
    dataset_path = os.path.join(PATH_ROOT,'dataset.pt')
    dataset = torch.load(dataset_path)

    index_dict = config["model::index_dict"]
    loss_recon_layer = LayerWiseReconLoss(
        reduction=config.get("training::reduction", "global_mean"),
        index_dict=index_dict,
        normalization_koeff=None,
        threshold=config.get("training::error_threshold", None),
    )

    def calc_loss(original, predicted, loss=loss_recon_layer):
        if len(original.shape) == 1:
            original = original.unsqueeze(0)
        if len(predicted.shape) == 1:
            predicted = predicted.unsqueeze(0)
        calc = loss(predicted.to(device), original.to(device))
        return calc

    weights_train = dataset['trainset'].__get_weights__()
    with torch.no_grad():
        _, y_unmodified_train = module.forward(weights_train.float().to(device))
        threshold = calc_loss(weights_train, y_unmodified_train)["loss_recon"]

    logger.info(f'Finished STEP 1: calculating threshold: {threshold}')

    weights_test = dataset['testset'].__get_weights__()
    weights_val = dataset['valset'].__get_weights__()
    train_size = len(weights_train)
    test_size = len(weights_test) + len(weights_val)
    
    weights = torch.cat([weights_train, weights_test, weights_val])

    avg_export = np.full((1), fill_value=-1.0)
    avg_export[0] = threshold
    losses_export = np.full((24, len(weights)), fill_value=-1.0)
    sizes_export = np.full((2), fill_value=-1.0)
    sizes_export[0] = train_size
    sizes_export[1] = test_size

    with torch.no_grad():
        _, y_unmodified_test = module.forward(weights.float().to(device))
        losses_unmodified = np.array([calc_loss(orig, pred)["loss_recon"].cpu() for orig, pred in zip(weights, y_unmodified_test)])

    losses_export[0,:] = losses_unmodified
    logger.info(f'Finished STEP 2: calculating unmodified zoo losses')

    weights_modified_z = ret_embedded_weights(zoo_name=zoo_name, malware_name=malware_name, lsb=-1, msb=msb, fill=True)
    def calc_modified(lsb=23):
        weights_modified = torch.from_numpy(weights_modified_z[str(lsb)])
        with torch.no_grad():
            _, y_modified = module.forward(weights_modified.float().to(device))
            losses_modified = np.array([calc_loss(orig, pred)["loss_recon"].cpu() for orig, pred in zip(weights_modified, y_modified)])
        losses_export[lsb,:] = losses_modified
    Parallel(n_jobs=1, prefer="threads")(delayed(calc_modified)(i) for i in range(1,24))

    logger.info(f'Finished STEP 3: calculating modified zoo losses')
    
    save_path = get_losses_path(zoo_name=zoo_name, malware_name=malware_name, fill=True, msb=msb)
    np.savez(save_path, avg=avg_export, losses=losses_export, sizes=sizes_export)

if __name__ == '__main__':
    create_grads_dataset(zoo_name='mnist', all_layers = False, malware_name = 'malware_12584bytes', msb = False, n_jobs = -1)