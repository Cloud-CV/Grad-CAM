from __future__ import absolute_import

from demo.celery import celery_app

import grad_cam.constants as constants

import os

@celery_app.task(ignore_result=True)
def grad_cam_classification(image_path, label, output_dir):
    print "Loading the model again"
    from grad_cam.torch_models import ClassificationTorchModel
    print "Successfully loaded the model again"
    classification_task = ClassificationTorchModel.predict(image_path, label, output_dir)
    print classification_task


# # @shared_task
# @celery_app.task(ignore_result=True)
# def grad_cam_captioning(image_path, caption, output_dir):

#     CaptioningTorchModel.predict(image_path, constants.VQA_CONFIG['input_sz'], constants.VQA_CONFIG['input_sz'], caption, output_dir)


# @shared_task
# def test():
#     return "The celery is working fine and returning the results"
