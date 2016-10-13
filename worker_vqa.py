from __future__ import absolute_import
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

import django
django.setup()

from grad_cam.models import VqaJob
from django.conf import settings
from grad_cam.utils import log_to_terminal

import grad_cam.constants as constants
import PyTorch
import PyTorchHelpers
import pika
import time
import yaml
import json
import traceback

# Loading the VQA Model forever
VQAModel = PyTorchHelpers.load_lua_class(constants.VQA_LUA_PATH, 'VQATorchModel')
VqaTorchModel = VQAModel(
    constants.VQA_CONFIG['proto_file'],
    constants.VQA_CONFIG['model_file'],
    constants.VQA_CONFIG['input_sz'],
    constants.VQA_CONFIG['backend'],
    constants.VQA_CONFIG['layer_name'],
    constants.VQA_CONFIG['model_path'],
    constants.VQA_CONFIG['input_encoding_size'],
    constants.VQA_CONFIG['rnn_size'],
    constants.VQA_CONFIG['rnn_layers'],
    constants.VQA_CONFIG['common_embedding_size'],
    constants.VQA_CONFIG['num_output'],
    constants.VQA_CONFIG['seed'],
    constants.VQA_GPUID,
)

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='vqa_task_queue', durable=True)
print(' [*] Waiting for messages. To exit press CTRL+C')

def callback(ch, method, properties, body):
    try:
        print(" [x] Received %r" % body)
        body = yaml.safe_load(body) # using yaml instead of json.loads since that unicodes the string in value

        result = VqaTorchModel.predict(body['image_path'], constants.VQA_CONFIG['input_sz'], constants.VQA_CONFIG['input_sz'], body['input_question'], body['input_answer'], body['output_dir'])

        VqaJob.objects.create(job_id=body['socketid'], input_answer=body['input_answer'], image=str(result['input_image']).replace(settings.BASE_DIR, '')[1:], predicted_answer = result['answer'], gcam_image=str(result['vqa_gcam']).replace(settings.BASE_DIR, '')[1:])

        result['input_image'] = str(result['input_image']).replace(settings.BASE_DIR, '')
        result['vqa_gcam'] = str(result['vqa_gcam']).replace(settings.BASE_DIR, '')
        result['vqa_gcam_raw'] = str(result['vqa_gcam_raw']).replace(settings.BASE_DIR, '')
        result['vqa_gb'] = str(result['vqa_gb']).replace(settings.BASE_DIR, '')
        result['vqa_gb_gcam'] = str(result['vqa_gb_gcam']).replace(settings.BASE_DIR, '')

        log_to_terminal(body['socketid'], {"terminal": json.dumps(result)})
        log_to_terminal(body['socketid'], {"result": json.dumps(result)})
        log_to_terminal(body['socketid'], {"terminal": "Completed the Grad-CAM VQA task"})

        ch.basic_ack(delivery_tag = method.delivery_tag)
    except Exception, err:
        log_to_terminal(body['socketid'], {"terminal": json.dumps({"Traceback": str(traceback.print_exc())})})

channel.basic_consume(callback,
                      queue='vqa_task_queue')

channel.start_consuming()
