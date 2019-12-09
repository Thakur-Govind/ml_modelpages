from django.db import models

# Create your models here.
class mlmodels(models.Model):
    image = models.ImageField(upload_to = 'images/')
    name = models.CharField(max_length = 100)
    para1 = models.CharField(max_length = 100, default = "")
    para2 = models.CharField(max_length = 100, default = "")
