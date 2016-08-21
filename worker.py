from __future__ import absolute_import

import os
import pika
import time

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

from django.conf import settings  # noqa

import grad_cam.constants as constants
import PyTorch
import PyTorchHelpers
import json
import yaml

ClassificationModel = PyTorchHelpers.load_lua_class(constants.CLASSIFICATION_LUA_PATH, 'ClassificationTorchModel')
ClassificationTorchModel = ClassificationModel(
    constants.CLASSIFICATION_CONFIG['proto_file'],
    constants.CLASSIFICATION_CONFIG['model_file'],
    constants.CLASSIFICATION_CONFIG['backend'],
    constants.CLASSIFICATION_CONFIG['input_sz'],
    constants.CLASSIFICATION_CONFIG['layer_name'],
    constants.CLASSIFICATION_CONFIG['seed'],
    settings.GPUID,
)

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='classify_task_queue', durable=True)
print(' [*] Waiting for messages. To exit press CTRL+C')

def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)
    body = yaml.safe_load(body) # using yaml instead of json.loads since that unicodes the string in value
    classification_task = ClassificationTorchModel.predict(body['image_path'], body['label'], body['output_dir'])
    print classification_task
    print(" [x] Done")
    ch.basic_ack(delivery_tag = method.delivery_tag)

channel.basic_consume(callback,
                      queue='classify_task_queue')

channel.start_consuming()