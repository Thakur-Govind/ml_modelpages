from django.db import models

# Create your models here.
class mlmodels(models.Model):
    image = models.ImageField(upload_to = 'images/')
    name = models.CharField(max_length = 100)
    para1 = models.CharField(max_length = 100, default = "")
    para2 = models.CharField(max_length = 100, default = "")
    class_or_reg = models.CharField(max_length =5, default = "")
class modelhist(models.Model):
    m_name = models.CharField(max_length = 100)
    m_para1 = models.CharField(max_length = 100, default = "")
    m_para2 = models.CharField(max_length = 100, default = "")
    f_score = models.DecimalField(max_digits = 5, decimal_places = 2)
    f_b_l_score = models.DecimalField(max_digits = 5, decimal_places = 2)
    f_b_h_score = models.DecimalField(max_digits = 5, decimal_places = 2)
    accuracy = models.DecimalField(max_digits = 5, decimal_places = 2)
    r_2_score = models.DecimalField(max_digits = 5, decimal_places = 2, default = 0.00)
    dataset_name = models.CharField(max_length = 1000)
class dataset(models.Model):
    d_name = models.CharField(default = 'Admission Data', max_length = 1000)
    d_fname = models.CharField(default = 'lr.csv', max_length = 1000)
