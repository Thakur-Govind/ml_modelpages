from django.contrib import admin
from django.urls import path,include
from django.conf import settings
from django.conf.urls.static import static
import mlmodels.views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', mlmodels.views.home, name = 'home'),
    path('model/', include('mlmodels.urls')),
    

]  + static(settings.MEDIA_URL, document_root= settings.MEDIA_ROOT)
