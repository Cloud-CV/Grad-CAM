from django.db import models
import uuid

class ClassificationJob(models.Model):
    job_id = models.CharField(max_length=1000, blank=True, null=True)
    image = models.CharField(max_length=1000, blank=True, null=True)
    input_label = models.CharField(max_length=1000, blank=True, null=True, default="")
    predicted_label = models.CharField(max_length=1000, blank=True, null=True)
    gcam_image = models.CharField(max_length=1000, blank=True, null=True)
    createdAt = models.DateTimeField("Time", null=True, auto_now_add=True)

    def __unicode__(self):
        return str(self.job_id)

class VqaJob(models.Model):
    job_id = models.CharField(max_length=1000, blank=True, null=True)
    image = models.CharField(max_length=1000, blank=True, null=True)
    input_answer = models.CharField(max_length=1000, blank=True, null=True, default="")
    predicted_answer = models.CharField(max_length=1000, blank=True, null=True)
    gcam_image = models.CharField(max_length=1000, blank=True, null=True)
    question = models.CharField(max_length=1000, blank=True, null=True)
    createdAt = models.DateTimeField("Time", null=True, auto_now_add=True)

    def __unicode__(self):
        return str(self.job_id)

class CaptioningJob(models.Model):
    job_id = models.CharField(max_length=1000, blank=True, null=True)
    image = models.CharField(max_length=1000, blank=True, null=True)
    input_caption = models.CharField(max_length=1000, blank=True, null=True, default="")
    predicted_caption = models.CharField(max_length=1000, blank=True, null=True)
    gcam_image = models.CharField(max_length=1000, blank=True, null=True)
    createdAt = models.DateTimeField("Time", null=True, auto_now_add=True)

    def __unicode__(self):
        return str(self.job_id)
