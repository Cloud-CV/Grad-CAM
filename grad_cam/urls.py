from django.conf.urls import patterns, include, url

urlpatterns = patterns('',
    # Examples:
    url(r'^vqa', 'grad_cam.views.vqa', name='vqa'),
    url(r'^$', 'grad_cam.views.home', name='home'),
)
