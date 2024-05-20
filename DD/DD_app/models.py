from django.db import models

# Create your models here.
class Files(models.Model):
    f_id=models.IntegerField(primary_key=True)
    filename=models.CharField(max_length=255)
    c_date=models.CharField(max_length=255)
    c_time=models.CharField(max_length=255)