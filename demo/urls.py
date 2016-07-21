from django.conf.urls import patterns, include, url
from django.contrib import admin
from django.conf import settings

from grad_cam.torch_models import VqaTorchModel

admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    url(r'^demo/', include('grad_cam.urls')),
    # url(r'^admin/', include(admin.site.urls)),
)


if settings.DEBUG:
    # static files (images, css, javascript, etc.)
    urlpatterns += patterns('',
        (r'^media/(?P<path>.*)$', 'django.views.static.serve', {
        'document_root': settings.MEDIA_ROOT}))
