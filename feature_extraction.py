import torch
import json
from pathlib import Path
import numpy as np
from ghrp.model_definitions.def_net import NNmodule
import os, sys
from joblib import Parallel, delayed

from data_locator import get_zoo_path, get_grads_path, ret_zoo_weights, ret_embedded_weights

import logging

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


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

    zoo_dir = get_zoo_path(zoo_name)
    PATH_ROOT = Path(zoo_dir)
    gpus = 1 if torch.cuda.is_available() else 0
    device = torch.device('cuda') if gpus>0 else torch.device('cpu')

    config_model_path = PATH_ROOT.joinpath('config_zoo.json')
    config_model = json.load(config_model_path.open('r'))
    model = NNmodule(config_model)
    model.to(device)

    #first_parameter = next(model.model.parameters())
    #input_shape = first_parameter.size()
    #input_shape = tuple(model.model.module_list[0].weight.size())[1:]
    #print(input_shape)

    inputs = torch.zeros((1,1,28,28))
    #inputs = torch.zeros(input_shape)
    labels = torch.zeros((1), dtype=int)

    inputs = inputs.to(device)
    labels = labels.to(device)
    layers = [model.model.module_list[-1]]
    
    zoo_weights = torch.from_numpy(ret_zoo_weights(zoo_name))

    exports = {}
    grads = create_grads(zoo_weights)
    exports['0'] = grads.cpu().numpy()
    logger.info(f'Finished creating grads: lsb:0')

    with ret_embedded_weights(zoo_name=zoo_name, malware_name=malware_name, lsb=-1, msb=msb, fill=True) as embedded_weights:
        Parallel(n_jobs=n_jobs, prefer="threads")(delayed(create_grads_and_export)(j) for j in range(23,0,-1))

    save_path = get_grads_path(zoo_name=zoo_name, malware_name=malware_name, fill=True, msb=msb)

    np.savez_compressed(save_path, **exports)

if __name__ == '__main__':
    create_grads_dataset(zoo_name='mnist', all_layers = False, malware_name = 'malware_12584bytes', msb = False, n_jobs = -1)