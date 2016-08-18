from __future__ import absolute_import
import os

from celery import Celery

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

from django.conf import settings  # noqa

import grad_cam.constants as constants

celery_app = Celery('demo',
                backend='redis://0.0.0.0:6379/0',
                broker='redis://0.0.0.0:6379/0',
                include=['grad_cam.tasks']
            )

# Using a string here means the worker will not have to
# pickle the object when using Windows.
celery_app.config_from_object('django.conf:settings')
celery_app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)
