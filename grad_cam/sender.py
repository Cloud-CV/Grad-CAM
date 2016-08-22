from django.conf import settings
from grad_cam.utils import log_to_terminal

import os
import pika
import sys
import json


def grad_cam_classification(image_path, label, out_dir, socketid):

    connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='classify_task_queue', durable=True)
    message = {
        'image_path': image_path,
        'label': label,
        'output_dir': out_dir,
        'socketid': socketid,
    }
    log_to_terminal(socketid, {"terminal": "Publishing job to Classification Queue"})
    channel.basic_publish(exchange='',
                      routing_key='classify_task_queue',
                      body=json.dumps(message),
                      properties=pika.BasicProperties(
                         delivery_mode = 2, # make message persistent
                      ))

    print(" [x] Sent %r" % message)
    log_to_terminal(socketid, {"terminal": "Job published successfully"})
    connection.close()


def grad_cam_vqa(input_question, input_answer, image_path, out_dir, socketid):
    connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='vqa_task_queue', durable=True)
    message = {
        'image_path': image_path,
        'input_question': input_question,
        'input_answer': input_answer,
        'output_dir': out_dir,
        'socketid': socketid,
    }
    log_to_terminal(socketid, {"terminal": "Publishing job to VQA Queue"})
    channel.basic_publish(exchange='',
                      routing_key='vqa_task_queue',
                      body=json.dumps(message),
                      properties=pika.BasicProperties(
                         delivery_mode = 2, # make message persistent
                      ))

    print(" [x] Sent %r" % message)
    log_to_terminal(socketid, {"terminal": "Job published successfully"})
    connection.close()
