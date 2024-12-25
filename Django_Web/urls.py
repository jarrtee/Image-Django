"""
URL configuration for Django_Web project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from dj_api.views import Dj_ApiViewSet, login, User_Data_ViewSet,User_Photo_ViewSet,update_data_view,WX_Login,WX_Basic_inf,WX_USER_MARK

router = routers.DefaultRouter()
router.register(r'dj_api', Dj_ApiViewSet)#注册视图集

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('login/', login),
    path('basicdata/', User_Data_ViewSet),
    path('basicphoto/', User_Photo_ViewSet),
    path('UpdateBasicData/', update_data_view),
    path('WX_Login/', WX_Login),
    path('WX_Basic_inf/', WX_Basic_inf),
    path('WX_USER_MARK/', WX_USER_MARK)
]
