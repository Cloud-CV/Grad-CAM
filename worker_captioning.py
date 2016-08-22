from __future__ import absolute_import
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

from django.conf import settings
from grad_cam.utils import log_to_terminal

import grad_cam.constants as constants
import PyTorch
import PyTorchHelpers
import pika
import time
import yaml
import json

CaptioningModel = PyTorchHelpers.load_lua_class(constants.CAPTIONING_LUA_PATH, 'CaptioningTorchModel')
CaptioningTorchModel = CaptioningModel(
    constants.CAPTIONING_CONFIG['model_path'],
    constants.CAPTIONING_CONFIG['backend'],
    constants.CAPTIONING_CONFIG['input_sz'],
    constants.CAPTIONING_CONFIG['layer'],
    constants.CAPTIONING_CONFIG['seed'],
    settings.GPUID,
)

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='captioning_task_queue', durable=True)
print(' [*] Waiting for messages. To exit press CTRL+C')

def callback(ch, method, properties, body):

    print(" [x] Received %r" % body)
    body = yaml.safe_load(body) # using yaml instead of json.loads since that unicodes the string in value

    result = CaptioningTorchModel.predict(body['image_path'], constants.VQA_CONFIG['input_sz'], constants.VQA_CONFIG['input_sz'], body['caption'], body['output_dir'])

    result['input_image'] = str(result['input_image']).replace(settings.BASE_DIR, '')
    result['captioning_gcam'] = str(result['captioning_gcam']).replace(settings.BASE_DIR, '')
    result['captioning_gcam_raw'] = str(result['captioning_gcam_raw']).replace(settings.BASE_DIR, '')
    result['captioning_gb'] = str(result['captioning_gb']).replace(settings.BASE_DIR, '')
    result['captioning_gb_gcam'] = str(result['captioning_gb_gcam']).replace(settings.BASE_DIR, '')

    log_to_terminal(body['socketid'], {"result": json.dumps(result)})
    log_to_terminal(body['socketid'], {"terminal": json.dumps(result)})
    log_to_terminal(body['socketid'], {"terminal": "Completed the Captioning job"})

    ch.basic_ack(delivery_tag = method.delivery_tag)

channel.basic_consume(callback,
                      queue='captioning_task_queue')

channel.start_consuming()
