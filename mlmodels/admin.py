from django.contrib import admin

# Register your models here.
from .models import mlmodels,modelhist,dataset
# Register your models here.
admin.site.register(mlmodels)
admin.site.register(modelhist)
admin.site.register(dataset)
