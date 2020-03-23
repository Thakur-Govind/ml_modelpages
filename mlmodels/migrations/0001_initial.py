# Generated by Django 2.0.2 on 2019-12-09 18:22

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='mlmodels',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='images/')),
                ('name', models.CharField(max_length=100)),
                ('para1', models.CharField(default='', max_length=100)),
                ('para2', models.CharField(default='', max_length=100)),
            ],
        ),
    ]