from channels import Group
import json

def log_to_terminal(socketid, message):
	Group(socketid).send({"text": json.dumps(message)})


# from django.conf import settings
# from grad_cam.torch_models import VqaTorchModel, ClassificationTorchModel, CaptioningTorchModel
# from grad_cam.torch_models import VqaTorchModel

# import grad_cam.constants as constants

# import PyTorch
# import PyTorchHelpers


# def grad_cam_vqa(input_question, input_answer, image_path, output_dir):

#     return VqaTorchModel.predict(image_path, constants.VQA_CONFIG['input_sz'], constants.VQA_CONFIG['input_sz'], input_question, input_answer, output_dir)


# def grad_cam_classification(image_path, label, output_dir):

#     return ClassificationTorchModel.predict(image_path, label, output_dir)


# def grad_cam_captioning(image_path, caption, output_dir):

#     return CaptioningTorchModel.predict(image_path, constants.VQA_CONFIG['input_sz'], constants.VQA_CONFIG['input_sz'], caption, output_dir)
