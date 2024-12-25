from rest_framework import serializers
from .models import Dj_Api


class Dj_ApiSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Dj_Api
        fields = "__all__"
