from django.db import models


class Dj_Api(models.Model):
    title = models.CharField(max_length=100)
    author = models.CharField(max_length=100)
    content = models.TextField()
    posttime = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'dj_api'


#数据库数据(建表)
class User_Data(models.Model):
    UserNum = models.CharField(max_length=255, null=False)
    UserName = models.CharField(max_length=255)
    PhoneNum = models.IntegerField()
    Picture = models.ImageField()

    def __str__(self):
        return f"['UserNum':{self.UserNum}, 'UserName':{self.UserName},'PhoneNum':{self.PhoneNum}, 'Picture':{self.Picture}]']"

    class Meta:
        db_table = 'user_basic_inf'


class User_Photo(models.Model):
    UserNum = models.CharField(max_length=255, null=False)
    Photo = models.ImageField()

    def __str__(self):
        return f"['UserNum':{self.UserNum}'Photo':{self.Photo}]']"

    class Meta:
        db_table = 'user_basic_photo'
