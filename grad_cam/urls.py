from django.conf.urls import patterns, include, url
from grad_cam import views

urlpatterns = patterns('',
    # Examples:
    url(r'^vqa', views.vqa, name='vqa'),
    url(r'^upload', views.file_upload, name='upload'),
    url(r'^classification', views.classification, name='classification'),
    url(r'^captioning', views.captioning, name='captioning'),
    url(r'^', views.home, name='home'),
)
