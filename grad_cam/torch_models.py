from django.conf import settings

import grad_cam.constants as constants

import PyTorch
import PyTorchHelpers


# Loading the classification model forever
ClassificationModel = PyTorchHelpers.load_lua_class(constants.CLASSIFICATION_LUA_PATH, 'ClassificationTorchModel')
ClassificationTorchModel = ClassificationModel(
    constants.CLASSIFICATION_CONFIG['proto_file'],
    constants.CLASSIFICATION_CONFIG['model_file'],
    constants.CLASSIFICATION_CONFIG['backend'],
    constants.CLASSIFICATION_CONFIG['input_sz'],
    constants.CLASSIFICATION_CONFIG['layer_name'],
    constants.CLASSIFICATION_CONFIG['seed'],
    settings.GPUID,
)


# Loading the VQA Model forever
# VQAModel = PyTorchHelpers.load_lua_class(constants.VQA_LUA_PATH, 'VQATorchModel')
# VqaTorchModel = VQAModel(
#     constants.VQA_CONFIG['proto_file'],
#     constants.VQA_CONFIG['model_file'],
#     constants.VQA_CONFIG['input_sz'],
#     constants.VQA_CONFIG['backend'],
#     constants.VQA_CONFIG['layer_name'],
#     constants.VQA_CONFIG['model_path'],
#     constants.VQA_CONFIG['input_encoding_size'],
#     constants.VQA_CONFIG['rnn_size'],
#     constants.VQA_CONFIG['rnn_layers'],
#     constants.VQA_CONFIG['common_embedding_size'],
#     constants.VQA_CONFIG['num_output'],
#     constants.VQA_CONFIG['seed'],
#     settings.GPUID,
# )


# Loading the Captioning model forever
CaptioningModel = PyTorchHelpers.load_lua_class(constants.CAPTIONING_LUA_PATH, 'CaptioningTorchModel')
CaptioningTorchModel = CaptioningModel(
    constants.CAPTIONING_CONFIG['model_path'],
    constants.CAPTIONING_CONFIG['backend'],
    constants.CAPTIONING_CONFIG['input_sz'],
    constants.CAPTIONING_CONFIG['layer'],
    constants.CAPTIONING_CONFIG['seed'],
    settings.GPUID,
)
