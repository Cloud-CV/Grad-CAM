from django.conf import settings

import grad_cam.constants as constants

import PyTorch
import PyTorchHelpers


# Loading the VQA Model forever
VQAModel = PyTorchHelpers.load_lua_class(constants.VQA_LUA_PATH, 'TorchModel')
VqaTorchModel = VQAModel(
    constants.VQA_CONFIG['proto_file'],
    constants.VQA_CONFIG['model_file'],
    constants.VQA_CONFIG['input_sz'],
    constants.VQA_CONFIG['backend'],
    constants.VQA_CONFIG['layer_name'],
    constants.VQA_CONFIG['model_path'],
    constants.VQA_CONFIG['input_encoding_size'],
    constants.VQA_CONFIG['rnn_size'],
    constants.VQA_CONFIG['rnn_layers'],
    constants.VQA_CONFIG['common_embedding_size'],
    constants.VQA_CONFIG['num_output'],
    constants.VQA_CONFIG['seed'],
    settings.GPUID,
)
