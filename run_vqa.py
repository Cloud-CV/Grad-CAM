import PyTorch
import PyTorchHelpers


def run():

    proto_file = 'models/VGG_ILSVRC_19_layers_deploy.prototxt'
    model_file = 'models/VGG_ILSVRC_19_layers.caffemodel'
    input_sz = 224
    backend = 'cudnn'
    layer_name = 'relu5_4'
    input_image_path = 'images/cat_dog.jpg'
    question = 'What animal?'
    answer = 'cat'
    model_path = 'VQA_LSTM_CNN/lstm.t7'
    input_encoding_size = 200
    rnn_size = 512
    rnn_layers = 2
    common_embedding_size = 1024 
    num_output = 1000
    seed = 123
    gpuid = 0
    out_path  = 'output/'

    TorchModel = PyTorchHelpers.load_lua_class('new_vqa.lua', 'TorchModel')
    torchModel = TorchModel(proto_file, model_file, input_sz, backend, layer_name, input_image_path, question, answer, model_path, input_encoding_size, rnn_size, rnn_layers, common_embedding_size, num_output, seed, gpuid, out_path)
    cnn = torchModel.predict(input_image_path, input_sz, input_sz)
    print cnn

run()
