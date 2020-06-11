# Generated by Django 2.2.9 on 2020-04-17 20:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mlmodels', '0002_dataset_modelhist'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='d_fname',
            field=models.CharField(default='lr.csv', max_length=1000),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='d_name',
            field=models.CharField(default='Admission Data', max_length=1000),
        ),
    ]