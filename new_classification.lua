require 'torch'
require 'nn'
require 'lfs'
require 'image'
require 'loadcaffe'
utils = require 'misc.utils'

local preprocess = utils.preprocess

local TorchModel = torch.class('TorchModel')

function TorchModel:__init(proto_file, model_file, backend, input_sz, layer_name, input_image_path, label, seed, gpuid, out_path)

  self:loadModel(proto_file, model_file, backend)
  self:processData(input_image_path, input_sz, input_sz, gpuid)

  self.proto_file = proto_file
  self.model_file = model_file
  self.backend = backend
  self.input_sz = input_sz
  self.layer_name = layer_name
  self.input_image_path = input_image_path
  self.label = label
  self.seed = seed
  self.gpuid = gpuid
  self.out_path = out_path

  torch.manualSeed(self.seed)
  torch.setdefaulttensortype('torch.DoubleTensor')
  lfs.mkdir(self.out_path)
  if gpuid >= 0 then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(gpuid + 1)
    cutorch.manualSeed(seed)
  end

end

function TorchModel:loadModel(proto_file, model_file, backend)

  self.net = loadcaffe.load(proto_file, model_file, backend)
  local cnn = self.net

  -- Set to evaluate and remove softmax layer
  cnn:evaluate()
  cnn:remove()

  -- Clone & replace ReLUs for Guided Backprop

end

function TorchModel:processData(input_image_path, input_sz, input_sz, gpuid)
  local cnn_gb = self.net:clone()

  cnn_gb:replace(utils.guidedbackprop)

  if gpuid >= 0 then
    self.net:cuda()
    cnn_gb:cuda()
    img = img:cuda()
  end

  local img = utils.preprocess(input_image_path, input_sz, input_sz)

  -- Forward pass
  local output = self.net:forward(img)
  local output_gb = cnn_gb:forward(img)

  -- Set gradInput
  local doutput = utils.create_grad_input(self.net.modules[#self.net.modules], self.label)

  -- Grad-CAM
  local result = {}
  local gcam = utils.grad_cam(self.net, self.layer_name, doutput)
  gcam = image.scale(gcam:float(), self.input_sz, self.input_sz)
  local hm = utils.to_heatmap(gcam)
  image.save(self.out_path .. 'classify_gcam_' .. self.label .. '.png', image.toDisplayTensor(hm))
  result[0] = self.out_path .. 'classify_gcam_' .. self.label .. '.png'
  -- Guided Backprop
  local gb_viz = cnn_gb:backward(img, doutput)
  image.save(self.out_path .. 'classify_gb_' .. self.label .. '.png', image.toDisplayTensor(gb_viz))
  result[1] = self.out_path .. 'classify_gb_' .. self.label .. '.png'
  -- Guided Grad-CAM
  local gb_gcam = gb_viz:float():cmul(gcam:expandAs(gb_viz))
  image.save(self.out_path .. 'classify_gb_gcam_' .. self.label .. '.png', image.toDisplayTensor(gb_gcam))
  result[2] = self.out_path .. 'classify_gb_gcam_' .. self.label .. '.png'

  return result
end
