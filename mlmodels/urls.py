from django.urls import path
from . import views
urlpatterns = [
    path('', views.home, name = 'mlhome'),
    path('params/<int:ml_id>/', views.params, name = 'mlparams'),
    path('/<int:ml_id>', views.mlexec, name = 'models'),

]
