def classification(image_path, label, output_dir):
    import pika
    import sys
    import json

    print "Job sent to the producer"
    connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='classify_task_queue', durable=True)

    message = {
        'image_path': image_path,
        'label': label,
        'output_dir': output_dir,
    }

    channel.basic_publish(exchange='',
                      routing_key='classify_task_queue',
                      body=json.dumps(message),
                      properties=pika.BasicProperties(
                         delivery_mode = 2, # make message persistent
                      ))

    print(" [x] Sent %r" % message)
    connection.close()
    return " [x] Sent %r" % message