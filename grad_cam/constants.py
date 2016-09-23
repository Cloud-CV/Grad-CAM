from django.conf import settings
import os

COCO_IMAGES_PATH = os.path.join(settings.MEDIA_ROOT, 'coco', 'val2014')

VQA_GPUID = 0

VQA_CONFIG = {
    'proto_file': 'models/VGG_ILSVRC_19_layers_deploy.prototxt',
    'model_file': 'models/VGG_ILSVRC_19_layers.caffemodel',
    'input_sz': 224,
    'backend': '',
    'layer_name': 'relu5_4',
    'model_path': 'VQA_LSTM_CNN/lstm.t7',
    'input_encoding_size': 200,
    'rnn_size': 512,
    'rnn_layers': 2,
    'common_embedding_size': 1024,
    'num_output': 1000,
    'seed': 123,
    'image_dir': os.path.join(settings.BASE_DIR, 'media', 'grad_cam', 'vqa')
}


if VQA_GPUID == -1:
    VQA_CONFIG['backend'] = "nn"
else:
    VQA_CONFIG['backend'] = "cudnn"

VQA_LUA_PATH = "visual_question_answering.lua"

CLASSIFICATION_GPUID = 1

CLASSIFICATION_CONFIG = {
    'proto_file': 'models/VGG_ILSVRC_16_layers_deploy.prototxt',
    'model_file': 'models/VGG_ILSVRC_16_layers.caffemodel',
    'input_sz': 224,
    'backend': 'cudnn',
    'layer_name': 'relu5_3',
    'seed': 123,
    'image_dir': os.path.join(settings.BASE_DIR, 'media', 'grad_cam', 'classification')
}


CLASSIFICATION_LUA_PATH = "classification.lua"

if CLASSIFICATION_GPUID == -1:
    CLASSIFICATION_CONFIG['backend'] = "nn"
else:
    CLASSIFICATION_CONFIG['backend'] = "cudnn"

CAPTIONING_GPUID = 2

CAPTIONING_CONFIG = {
    'input_sz': 224,
    'backend': 'cudnn',
    'layer': 30,
    'model_path': 'neuraltalk2/model_id1-501-1448236541.t7',
    'seed': 123,
    'image_dir': os.path.join(settings.BASE_DIR, 'media', 'grad_cam', 'captioning')
}

CAPTIONING_LUA_PATH = "captioning.lua"

if CAPTIONING_GPUID == -1:
    CAPTIONING_CONFIG['backend'] = "nn"
else:
    CAPTIONING_CONFIG['backend'] = "cudnn"
