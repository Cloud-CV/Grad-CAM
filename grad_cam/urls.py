from django.conf.urls import patterns, include, url

urlpatterns = patterns('',
    # Examples:
    # url(r'^vqa', 'grad_cam.views.vqa', name='vqa'),
    url(r'^upload', 'grad_cam.views.file_upload', name='upload'),
    url(r'^classification', 'grad_cam.views.classification', name='classification'),
    url(r'^captioning', 'grad_cam.views.captioning', name='captioning'),
    # url(r'^$', 'grad_cam.views.home', name='home'),
    # url(r'^vqa/test', 'grad_cam.views.vqa_new', name='vqa_new'),
    # url(r'^vqa/upload', 'grad_cam.views.vqa_upload', name='vqa_upload'),
    # url(r'^test', 'grad_cam.views.home_new', name='home_new'),
    # url(r'^classification/upload', 'grad_cam.views.classification_upload', name='classification'),
    # url(r'^captioning/upload', 'grad_cam.views.captioning_upload', name='captioning_upload'),
)
