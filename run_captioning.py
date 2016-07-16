import PyTorch
import PyTorchHelpers



def run():

    model_path = "neuraltalk2/model_id1-501-1448236541.t7"
    backend = "cudnn"
    input_sz = 224
    layer = 30
    input_image_path = 'images/cat_dog.jpg'
    caption = 'a dog and a cat posing for a picture'
    seed = 123
    gpuid = 0
    out_path = 'output/'

    TorchModel = PyTorchHelpers.load_lua_class('new_captioning.lua', 'TorchModel')
    torchModel = TorchModel(model_path, backend, input_sz, layer, input_image_path, caption, seed, gpuid, out_path)
    cnn = torchModel.predict(input_image_path, input_sz, input_sz)
    print "The final result is ", cnn

run()
