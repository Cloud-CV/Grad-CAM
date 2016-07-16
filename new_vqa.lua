require 'torch'
require 'nn'
require 'lfs'
require 'image'
require 'loadcaffe'
utils = require 'misc.utils'

local preprocess = utils.preprocess

local TorchModel = torch.class('TorchModel')

function TorchModel:__init(proto_file, model_file, input_sz, backend, layer_name, input_image_path, question, answer, model_path, input_encoding_size, rnn_size, rnn_layers, common_embedding_size, num_output, seed, gpuid, out_path )

  self.proto_file = proto_file
  self.model_file = model_file
  self.input_sz = input_sz
  self.backend = backend
  self.layer_name = layer_name
  self.input_image_path = input_image_path
  self.question = question
  self.answer = answer
  self.model_path = model_path
  self.input_encoding_size = input_encoding_size
  self.rnn_size = rnn_size
  self.rnn_layers = rnn_layers
  self.common_embedding_size = common_embedding_size
  self.num_output = num_output
  self.seed = seed
  self.gpuid = gpuid
  self.out_path = out_path
  self:loadModel(proto_file, model_file, backend)

  torch.manualSeed(self.seed)
  torch.setdefaulttensortype('torch.FloatTensor')
  lfs.mkdir(self.out_path)

  if self.gpuid >= 0 then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(self.gpuid + 1)
    cutorch.manualSeed(self.seed)
  end

end

function TorchModel:loadModel(proto_file, model_file, backend)
  -- Load CNN
  self.net = loadcaffe.load(proto_file, model_file, backend)

  -- Set to evaluate and remove linear+softmax layer
  self.net:evaluate()
  self.net:remove()
  self.net:remove()
  self.net:add(nn.Normalize(2))

  -- Clone & replace ReLUs for Guided Backprop
  local cnn_gb = self.net:clone()
  cnn_gb:replace(utils.guidedbackprop)
  self.cnn_gb = cnn_gb
  -- VQA-specific dependencies
  -- https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/eval.lua

  -- Below is a hacky solution since opt.gpuid is used in VQA_LSTM_CNN/misc.RNNUtils 
  opt = {}
  opt.gpuid = self.gpuid

  require 'VQA_LSTM_CNN/misc.netdef'
  require 'VQA_LSTM_CNN/misc.RNNUtils'
  LSTM = require 'VQA_LSTM_CNN/misc.LSTM'
  cjson = require 'cjson'

  -- Load vocabulary
  local file = io.open('VQA_LSTM_CNN/data_prepro.json','r')
  local text = file:read()
  file:close()
  local json_file = cjson.decode(text)
  local vocabulary_size_q = 0
  for i, w in pairs(json_file['ix_to_word']) do vocabulary_size_q = vocabulary_size_q + 1 end


  -- VQA model definition
  local embedding_net_q = nn.Sequential()
    :add(nn.Linear(vocabulary_size_q, self.input_encoding_size))
    :add(nn.Dropout(0.5))
    :add(nn.Tanh())

  local encoder_net_q = LSTM.lstm_conventional(self.input_encoding_size, self.rnn_size, 1, self.rnn_layers, 0.5)

  local multimodal_net = nn.Sequential()
    :add(netdef.AxB(2 * self.rnn_size * self.rnn_layers, 4096, self.common_embedding_size, 0.5))
    :add(nn.Dropout(0.5))
    :add(nn.Linear(self.common_embedding_size, self.num_output))

  local dummy_state_q = torch.Tensor(self.rnn_size * self.rnn_layers * 2):fill(0)
  local dummy_output_q = torch.Tensor(1):fill(0)

  -- Ship model to GPU
  if self.gpuid >= 0 then
    embedding_net_q:cuda()
    encoder_net_q:cuda()
    multimodal_net:cuda()
    dummy_state_q = dummy_state_q:cuda()
    dummy_output_q = dummy_output_q:cuda()
  end

  -- Set to evaluate
  embedding_net_q:evaluate()
  encoder_net_q:evaluate()
  multimodal_net:evaluate()

  -- Zero gradients
  embedding_net_q:zeroGradParameters()
  encoder_net_q:zeroGradParameters()
  multimodal_net:zeroGradParameters()

  -- Load pretrained VQA model
  embedding_w_q, embedding_dw_q = embedding_net_q:getParameters()
  encoder_w_q, encoder_dw_q = encoder_net_q:getParameters()
  multimodal_w, multimodal_dw = multimodal_net:getParameters()

  model_param = torch.load(self.model_path)
  embedding_w_q:copy(model_param['embedding_w_q'])
  encoder_w_q:copy(model_param['encoder_w_q'])
  multimodal_w:copy(model_param['multimodal_w'])

  local encoder_net_buffer_q = dupe_rnn(encoder_net_q, 26)

  -- all below variables are used in predict method
  self.embedding_net_q = embedding_net_q
  self.encoder_net_buffer_q = encoder_net_buffer_q
  self.vocabulary_size_q = vocabulary_size_q
  self.multimodal_net = multimodal_net
  self.dummy_state_q = dummy_state_q
  self.json_file = json_file

end

function TorchModel:predict(input_image_path, input_sz, input_sz)

  -- Load image
  local img = utils.preprocess(input_image_path, input_sz, input_sz)
  -- Ship CNNs and image to GPU
  if self.gpuid >= 0 then
    self.net:cuda()
    self.cnn_gb:cuda()
    img = img:cuda()
  end

  -- Forward pass
  fv_im = self.net:forward(img)
  fv_im_gb = self.cnn_gb:forward(img)

  -- Tokenize question
  local cmd = 'python misc/prepro_ques.py --question "'.. self.question..'"'
  os.execute(cmd)
  local file = io.open('ques_feat.json')
  local text = file:read()
  file:close()
  q_feats = cjson.decode(text)

  question = right_align(torch.LongTensor{q_feats.ques}, torch.LongTensor{q_feats.ques_length})
  fv_sorted_q = sort_encoding_onehot_right_align(question, torch.LongTensor{q_feats.ques_length}, self.vocabulary_size_q)

  -- Ship question features to GPU
  if self.gpuid >= 0 then
    fv_sorted_q[1] = fv_sorted_q[1]:cuda()
    fv_sorted_q[3] = fv_sorted_q[3]:cuda()
    fv_sorted_q[4] = fv_sorted_q[4]:cuda()
  end

  local question_max_length = fv_sorted_q[2]:size(1)

  -- Embedding forward
  local word_embedding_q = split_vector(self.embedding_net_q:forward(fv_sorted_q[1]), fv_sorted_q[2])

  -- Encoder forward
  local states_q, _ = rnn_forward(self.encoder_net_buffer_q, torch.repeatTensor(self.dummy_state_q:fill(0), 1, 1), word_embedding_q, fv_sorted_q[2])

  -- Multimodal forward
  local tv_q = states_q[question_max_length + 1]:index(1, fv_sorted_q[4])
  local scores = self.multimodal_net:forward({tv_q, fv_im})

  -- Get predictions
  _, pred = torch.max(scores:double(), 2)
  answer = self.json_file['ix_to_ans'][tostring(pred[{1, 1}])]

  local inv_vocab = utils.table_invert(self.json_file['ix_to_ans'])
  if self.answer ~= '' then answer_idx = inv_vocab[self.answer] else self.answer = answer answer_idx = inv_vocab[answer] end

  print("Question: ", self.question)
  print("Predicted answer: ", answer)
  print("Grad-CAM answer: ", self.answer)

  -- Set gradInput
  local doutput = utils.create_grad_input(self.multimodal_net.modules[#self.multimodal_net.modules], answer_idx)

  -- Multimodal backward
  local tmp = self.multimodal_net:backward({tv_q, fv_im}, doutput:view(1,-1))
  local dcnn = tmp[2]

  -- Grad-CAM
  local gcam = utils.grad_cam(self.net, self.layer_name, dcnn)
  gcam = image.scale(gcam:float(), self.input_sz, self.input_sz)

  local result = {}
  local hm = utils.to_heatmap(gcam)
  image.save(self.out_path .. 'vqa_gcam_' .. self.answer .. '.png', image.toDisplayTensor(hm))
  result[0] = self.out_path .. 'vqa_gcam_' .. self.answer .. '.png'

  -- Guided Backprop
  local gb_viz = self.cnn_gb:backward(img, dcnn)
  image.save(self.out_path .. 'vqa_gb_' .. self.answer .. '.png', image.toDisplayTensor(gb_viz))
  result[1] = self.out_path .. 'vqa_gb_' .. self.answer .. '.png'

  -- Guided Grad-CAM
  local gb_gcam = gb_viz:float():cmul(gcam:expandAs(gb_viz))
  image.save(self.out_path .. 'vqa_gb_gcam_' .. self.answer .. '.png', image.toDisplayTensor(gb_gcam))
  result[2] = self.out_path .. 'vqa_gb_gcam_' .. self.answer .. '.png'

  return result
end
