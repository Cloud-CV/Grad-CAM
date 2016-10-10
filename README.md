
# Grad-CAM: Gradient-weighted Class Activation Mapping

Grad-CAM uses the class-specific gradient information flowing into the final convolutional layer of a CNN to produce a coarse localization map of the important regions in the image. It is a novel technique for making CNN more 'transparent' by producing **visual explanations** i.e visualizations showing what evidence in the image supports a prediction. You can play with Grad-CAM demonstrations at the following links:

VQA Demo: http://gradcam.cloudcv.org/vqa

![Imgur](http://i.imgur.com/6jB4lAq.gif)

Classification Demo: http://gradcam.cloudcv.org/vqa

![Imgur](http://i.imgur.com/a1IiQg4.gif)

Captioning Demo: http://gradcam.cloudcv.org/vqa

![Imgur](http://i.imgur.com/BsOOpIn.gif)

## Installing / Getting started

We use RabbitMQ to queue the submitted jobs. Also, we use Redis as backend for realtime communication using websockets.

All the instructions for setting Grad-CAM from scratch can be found  [here](http://github.com/Cloud-CV/Grad-CAM/installation.md)

Note: For best results, its recommended to run the Grad-CAM demo on GPU enabled machines.

## Interested in Contributing?

Cloud-CV always welcomes new contributors to learn the new cutting edge technologies. If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

if you have more questions about the project, then you can talk to us on our [Gitter Channel](https://gitter.im/Cloud-CV/Grad-CAM).  

## Acknowledgements

- [VQA_LSTM_CNN](https://github.com/VT-vision-lab/VQA_LSTM_CNN)
- [HieCoAttenVQA](https://github.com/jiasenlu/HieCoAttenVQA)
- [NeuralTalk2](https://github.com/karpathy/neuraltalk2/)
- [PyTorch](https://github.com/hughperkins/pytorch)
