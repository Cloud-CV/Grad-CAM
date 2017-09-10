require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'
utils = require 'misc.utils'

local preprocess = utils.preprocess

local TorchModel = torch.class('ClassificationTorchModel')

function TorchModel:__init(proto_file, model_file, backend, input_sz, layer_name, seed, gpuid)

  self.proto_file = proto_file
  self.model_file = model_file
  self.backend = backend
  self.input_sz = input_sz
  self.layer_name = layer_name
  self.seed = seed
  self.gpuid = gpuid
  self:loadModel(proto_file, model_file, backend)
  torch.manualSeed(self.seed)
  -- GPU
  -- torch.setdefaulttensortype('torch.DoubleTensor')
  -- CPU
  torch.setdefaulttensortype('torch.FloatTensor')
  
  if gpuid >= 0 then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(1)
    cutorch.manualSeed(seed)
  end

end

function TorchModel:loadModel(proto_file, model_file, backend)

  self.net = loadcaffe.load(proto_file, model_file, backend)
  self.net = self.net:float()

  -- Set to evaluate and remove softmax layer
  self.net:evaluate()
  self.net:remove()

  -- Create GB CNN here itself
  self.net_gb = self.net:clone()
  self.net_gb:replace(utils.guidedbackprop)
  self.net_gb = self.net_gb:float()
end

function TorchModel:predict(input_image_path, label, out_path)
  local img = utils.preprocess(input_image_path, input_sz, input_sz)

  if self.gpuid >= 0 then
    self.net:cuda()
    self.net_gb:cuda()
    img = img:cuda()
  else
    print("THIS GOT EXECUTED")
    img = img:float()
  end
  
  -- Forward pass
  local output = self.net:forward(img)
  local output_gb = self.net_gb:forward(img)

  -- Take argmax
  local score, pred_label = torch.max(output,1)

  if label == -1 then 
    print("No label provided, using predicted label ", pred_label:float())
    label = pred_label[1]
  end

  -- Set gradInput
  local doutput = utils.create_grad_input(self.net.modules[#self.net.modules], label)

  -- Grad-CAM
  local result = {}
  local gcam = utils.grad_cam(self.net, self.layer_name, doutput)
  gcam = image.scale(gcam:float(), self.input_sz, self.input_sz)
  local hm = utils.to_heatmap(gcam)

  image.save(out_path .. 'classify_gcam_raw_' .. label .. '.png', image.toDisplayTensor(gcam))
  result['classify_gcam_raw'] = out_path .. 'classify_gcam_raw_' .. label .. '.png'

  image.save(out_path .. 'classify_gcam_' .. label .. '.png', image.toDisplayTensor(hm))
  result['classify_gcam'] = out_path .. 'classify_gcam_' .. label .. '.png'

  -- Guided Backprop
  local gb_viz = self.net_gb:backward(img, doutput)
  
  -- BGR to RGB
  gb_viz = gb_viz:index(1, torch.LongTensor{3, 2, 1})
  image.save(out_path .. 'classify_gb_' .. label .. '.png', image.toDisplayTensor(gb_viz))
  result['classify_gb'] = out_path .. 'classify_gb_' .. label .. '.png'

  -- Guided Grad-CAM
  local gb_gcam = gb_viz:float():cmul(gcam:expandAs(gb_viz))
  image.save(out_path .. 'classify_gb_gcam_' .. label .. '.png', image.toDisplayTensor(gb_gcam))
  result['classify_gb_gcam'] = out_path .. 'classify_gb_gcam_' .. label .. '.png'
  result['input_image'] = input_image_path

  result['label'] = label
  result['pred_label'] = pred_label[1]
  return result

end
