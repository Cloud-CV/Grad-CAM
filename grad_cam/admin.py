from django.contrib import admin

# Register your models here.
from grad_cam.models import VqaJob, ClassificationJob, CaptioningJob


class VqaJobAdmin(admin.ModelAdmin):
    list_display = ('job_id', 'image_url', 'question', 'input_answer', 'predicted_answer', 'gcam_image_url', 'createdAt')

    def image_url(self, obj):
        return '<img src="%s" alt="%s" height="150px">' % (obj.image, obj.image)
    image_url.allow_tags = True


    def gcam_image_url(self, obj):
        return '<img src="%s" alt="%s" height="150px">' % (obj.gcam_image, obj.gcam_image)
    gcam_image_url.allow_tags = True

class ClassificationJobAdmin(admin.ModelAdmin):
    list_display = ('job_id', 'image_url', 'input_label', 'predicted_label', 'gcam_image_url', 'createdAt')

    def image_url(self, obj):
        return '<img src="%s" alt="%s" height="150px">' % (obj.image, obj.image)
    image_url.allow_tags = True


    def gcam_image_url(self, obj):
        return '<img src="%s" alt="%s" height="150px">' % (obj.gcam_image, obj.gcam_image)
    gcam_image_url.allow_tags = True

class CaptioningJobAdmin(admin.ModelAdmin):
    list_display = ('job_id', 'show_image_url', 'input_caption', 'predicted_caption', 'show_gcam_image_url', 'createdAt')

    def show_image_url(self, obj):
        return '<img src="%s" alt="%s" height="150px">' % (obj.image, obj.image)
    show_image_url.allow_tags = True


    def show_gcam_image_url(self, obj):
        return '<img src="%s" alt="%s" height="150px">' % (obj.gcam_image, obj.gcam_image)
    show_gcam_image_url.allow_tags = True

admin.site.register(VqaJob, VqaJobAdmin)
admin.site.register(ClassificationJob, ClassificationJobAdmin)
admin.site.register(CaptioningJob, CaptioningJobAdmin)
