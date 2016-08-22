from __future__ import absolute_import
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

from django.conf import settings

import grad_cam.constants as constants
import PyTorch
import PyTorchHelpers
import pika
import time
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

    response = ClassificationTorchModel.predict(body['image_path'], body['label'], body['output_dir'])

    response['input_image'] = str(response['input_image']).replace(settings.BASE_DIR, '')
    response['classify_gcam'] = str(response['classify_gcam']).replace(settings.BASE_DIR, '')
    response['classify_gcam_raw'] = str(response['classify_gcam_raw']).replace(settings.BASE_DIR, '')
    response['classify_gb'] = str(response['classify_gb']).replace(settings.BASE_DIR, '')
    response['classify_gb_gcam'] = str(response['classify_gb_gcam']).replace(settings.BASE_DIR, '')

    print response

    log_to_terminal(body['socketid'], {"terminal": "Completed the Classification Task"})
    log_to_terminal(body['socketid'], {"result": classification_result})

    ch.basic_ack(delivery_tag = method.delivery_tag)

channel.basic_consume(callback,
                      queue='classify_task_queue')

channel.start_consuming()
