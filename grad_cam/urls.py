from django.conf.urls import patterns, include, url

urlpatterns = patterns('',
    # Examples:
    # url(r'^test', 'grad_cam.views.home_new', name='home_new'),
    url(r'^upload', 'grad_cam.views.file_upload', name='upload'),
    url(r'^classification', 'grad_cam.views.classification', name='classification'),
    url(r'^classification/upload', 'grad_cam.views.classification_upload', name='classification_upload'),
    # url(r'^$', 'grad_cam.views.home', name='home'),
)
