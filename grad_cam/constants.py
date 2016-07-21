from django.conf import settings
import os

GRAD_CAM_DEMO_IMAGES_PATH = os.path.join(settings.BASE_DIR, 'media', 'grad_cam', 'demo')

COCO_IMAGES_PATH = os.path.join(settings.MEDIA_URL, 'grad_cam', 'demo')

VQA_CONFIG = {
    'proto_file': 'models/VGG_ILSVRC_19_layers_deploy.prototxt',
    'model_file': 'models/VGG_ILSVRC_19_layers.caffemodel',
    'input_sz': 224,
    'backend': 'cudnn',
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

VQA_LUA_PATH = "new_vqa.lua"


# CLASSIFICATION_CONFIG = {
#     'proto_file': 'models/VGG_ILSVRC_16_layers_deploy.prototxt',
#     'model_file': 'models/VGG_ILSVRC_16_layers.caffemodel',
#     'input_sz': 224,
#     'backend': 'cudnn',
#     'layer_name': 'relu5_3',
#     'seed': 123,
#     'image_dir': os.path.join(settings.BASE_DIR, 'media', 'grad_cam', 'classification')
# }

# CLASSIFICATION_LUA_PATH = "new_classification.lua"


# CAPTIONING_CONFIG = {
#     'input_sz': 224,
#     'backend': 'cudnn',
#     'layer': 30,
#     'model_path': 'neuraltalk2/model_id1-501-1448236541.t7',
#     'seed': 123,
#     'image_dir': os.path.join(settings.BASE_DIR, 'media', 'grad_cam', 'captioning')
# }

# CAPTIONING_LUA_PATH = "new_captioning.lua"
