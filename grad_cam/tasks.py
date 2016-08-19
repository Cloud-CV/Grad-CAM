from __future__ import absolute_import

from demo.celery import celery_app

import grad_cam.constants as constants

import os
# from grad_cam.torch_models import VqaTorchModel, ClassificationTorchModel, CaptioningTorchModel
try:
    is_worker = os.environ['celery_worker']
    print "ENV VARIABLE CELERY_WORKER IS ", os.environ['celery_worker']
    # print os.environ['LD_LIBRARY_PATH'] 
    print "############## LOADING THE MODELS IN CELERY WORKER ################"
    from grad_cam.torch_models import ClassificationTorchModel, CaptioningTorchModel
    print "############## SUCCESSFULLY LOADED THE MODELS IN CELERY WORKER ################"
except Exception as e:
    print str(e)
    print "??????????? SOME ERROR OCCURRED ??????????????"


# @celery_app.task(ignore_result=True)
# def grad_cam_vqa(input_question, input_answer, image_path, output_dir, VqaTorchModel):
#     vqa_task_response = VqaTorchModel.predict(
#                         image_path, constants.VQA_CONFIG['input_sz'],
#                         constants.VQA_CONFIG['input_sz'],
#                         input_question, input_answer,
#                         output_dir,
#                     )
#     print "The response of vqa task is", vqa_task_response
#     return vqa_task_response


# @shared_task
@celery_app.task(ignore_result=True)
def grad_cam_classification(image_path, label, output_dir):
    try:
        print os.environ['LD_LIBRARY_PATH']
    except Exception as e:
        print str(e)
    classification_task = ClassificationTorchModel.predict(image_path, label, output_dir)
    print classification_task


# @shared_task
@celery_app.task(ignore_result=True)
def grad_cam_captioning(image_path, caption, output_dir):

    CaptioningTorchModel.predict(image_path, constants.VQA_CONFIG['input_sz'], constants.VQA_CONFIG['input_sz'], caption, output_dir)


# @shared_task
# def test():
#     return "The celery is working fine and returning the results"
