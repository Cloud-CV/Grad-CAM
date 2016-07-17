require 'torch'
require 'nn'
require 'lfs'
require 'image'
utils = require 'misc.utils'

local preprocess = utils.preprocess

local TorchModel = torch.class('TorchModel')

function TorchModel:__init(model_path, backend, input_sz, layer, seed, gpuid, out_path)

  self.model_path = model_path
  self.backend = backend
  self.input_sz = input_sz
  self.layer = layer
  self.seed = seed
  self.gpuid = gpuid
  self.out_path = out_path

  torch.manualSeed(self.seed)
  torch.setdefaulttensortype('torch.FloatTensor')
  lfs.mkdir(self.out_path)

  if self.gpuid >= 0 then
    require 'cunn'
    require 'cudnn'
    require 'cutorch'
    cutorch.setDevice(self.gpuid + 1)
    cutorch.manualSeed(self.seed)
  end

  -- neuraltalk2-specific dependencies
  -- https://github.com/karpathy/neuraltalk2

  self:loadModel(model_path)

end


function TorchModel:loadModel(model_path)

  -- Load the models
  local lm_misc_utils = require 'neuraltalk2.misc.utils'
  require 'neuraltalk2.misc.LanguageModel'
  local net_utils = require 'neuraltalk2.misc.net_utils'

  self.net = torch.load(model_path)
  local cnn_lm_model = self.net
  local cnn = cnn_lm_model.protos.cnn
  local lm = cnn_lm_model.protos.lm
  local vocab = cnn_lm_model.vocab

  net_utils.unsanitize_gradients(cnn)
  local lm_modules = lm:getModulesList()
  for k,v in pairs(lm_modules) do
    net_utils.unsanitize_gradients(v)
  end

  -- Set to evaluate mode
  lm:evaluate()
  cnn:evaluate()
  self.cnn = cnn
  self.lm = lm
  self.net_utils = net_utils
  self.vocab = vocab

end


function TorchModel:predict(input_image_path, input_sz, input_sz, input_caption)

  local img = utils.preprocess(input_image_path, input_sz, input_sz)

  -- Clone & replace ReLUs for Guided Backprop
  local cnn_gb = self.cnn:clone()
  cnn_gb:replace(utils.guidedbackprop)

  -- Ship model to GPU
  if self.gpuid >= 0 then
    self.cnn:cuda()
    cnn_gb:cuda()
    img = img:cuda()
    self.lm:cuda()
  end

  -- Forward pass
  im_feats = self.cnn:forward(img)
  im_feat = im_feats:view(1, -1)
  im_feat_gb = cnn_gb:forward(img)

  -- get the prediction from model
  local seq, seqlogps = self.lm:sample(im_feat, sample_opts)
  seq[{{}, 1}] = seq

  local caption = self.net_utils.decode_sequence(self.vocab, seq)

  if input_caption == '' then
    print("No caption provided, using generated caption for Grad-CAM.")
    input_caption = caption[1]
  end

  print("Generated caption: ", caption[1])
  print("Grad-CAM caption: ", input_caption)

  local seq_length = self.seq_length or 16

  local labels = utils.sent_to_label(self.vocab, input_caption, seq_length)
  if self.gpuid >=0 then labels = labels:cuda() end

  local logprobs = self.lm:forward({im_feat, labels})

  local doutput = utils.create_grad_input_lm(logprobs, labels)
  if self.gpuid >=0 then doutput = doutput:cuda() end

  -- lm backward
  local dlm, ddummy = unpack(self.lm:backward({im_feat, labels}, doutput))
  local dcnn = dlm[1]

  -- Grad-CAM
  local gcam = utils.grad_cam(self.cnn, self.layer, dcnn)
  gcam = image.scale(gcam:float(), self.input_sz, self.input_sz)

  local result = {}
  local hm = utils.to_heatmap(gcam)
  image.save(self.out_path .. 'caption_gcam_'  .. input_caption .. '.png', image.toDisplayTensor(hm))
  result[0] = self.out_path .. 'caption_gcam_'  .. input_caption .. '.png'

  -- Guided Backprop
  local gb_viz = cnn_gb:backward(img, dcnn)
  image.save(self.out_path .. 'caption_gb_' .. input_caption .. '.png', image.toDisplayTensor(gb_viz))
  result[1] = self.out_path .. 'caption_gb_' .. input_caption .. '.png'

  -- Guided Grad-CAM
  local gb_gcam = gb_viz:float():cmul(gcam:expandAs(gb_viz))
  image.save(self.out_path .. 'caption_gb_gcam_' .. input_caption .. '.png', image.toDisplayTensor(gb_gcam))
  result[2] = self.out_path .. 'caption_gb_gcam_' .. input_caption .. '.png'
  return result

end
