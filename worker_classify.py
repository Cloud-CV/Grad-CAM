from __future__ import absolute_import
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

import django
django.setup()


from django.conf import settings
from grad_cam.utils import log_to_terminal
from grad_cam.models import ClassificationJob
import grad_cam.constants as constants
import PyTorch
import PyTorchHelpers
import pika
import time
import yaml
import json
import traceback

# Close the database connection in order to make sure that MYSQL Timeout doesn't occur
django.db.close_old_connections()

ClassificationModel = PyTorchHelpers.load_lua_class(constants.CLASSIFICATION_LUA_PATH, 'ClassificationTorchModel')
ClassificationTorchModel = ClassificationModel(
    constants.CLASSIFICATION_CONFIG['proto_file'],
    constants.CLASSIFICATION_CONFIG['model_file'],
    constants.CLASSIFICATION_CONFIG['backend'],
    constants.CLASSIFICATION_CONFIG['input_sz'],
    constants.CLASSIFICATION_CONFIG['layer_name'],
    constants.CLASSIFICATION_CONFIG['seed'],
    constants.CLASSIFICATION_GPUID,
)

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='classify_task_queue', durable=True)
print(' [*] Waiting for messages. To exit press CTRL+C')

def callback(ch, method, properties, body):
    try:
        print(" [x] Received %r" % body)
        body = yaml.safe_load(body) # using yaml instead of json.loads since that unicodes the string in value

        result = ClassificationTorchModel.predict(body['image_path'], body['label'], body['output_dir'])

        ClassificationJob.objects.create(job_id=body['socketid'], input_label=body['label'], image=str(result['input_image']).replace(settings.BASE_DIR, '')[1:], predicted_label = result['pred_label'], gcam_image=str(result['classify_gcam']).replace(settings.BASE_DIR, '')[1:])

        # Close the database connection in order to make sure that MYSQL Timeout doesn't occur
        django.db.close_old_connections()

        result['input_image'] = str(result['input_image']).replace(settings.BASE_DIR, '')
        result['classify_gcam'] = str(result['classify_gcam']).replace(settings.BASE_DIR, '')
        result['classify_gcam_raw'] = str(result['classify_gcam_raw']).replace(settings.BASE_DIR, '')
        result['classify_gb'] = str(result['classify_gb']).replace(settings.BASE_DIR, '')
        result['classify_gb_gcam'] = str(result['classify_gb_gcam']).replace(settings.BASE_DIR, '')

        print result

        log_to_terminal("Hello", {"terminal": "Completed the Classification Task"})
        log_to_terminal(body['socketid'], {"terminal": json.dumps(result)})
        log_to_terminal(body['socketid'], {"result": json.dumps(result)})
        log_to_terminal(body['socketid'], {"terminal": "Completed the Classification Task"})

        ch.basic_ack(delivery_tag = method.delivery_tag)
    except Exception, err:
        log_to_terminal(body['socketid'], {"terminal": json.dumps({"Traceback": str(traceback.print_exc())})})

channel.basic_consume(callback,
                      queue='classify_task_queue')

channel.start_consuming()
