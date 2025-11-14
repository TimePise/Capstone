from django.db import models
from django.utils import timezone

class FallAlert(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    message = models.CharField(max_length=200)
    part = models.CharField(max_length=20)
    fall_level = models.CharField(max_length=10, default="심각")  # 추가
    name = models.CharField(max_length=50, default="환자A")       # 추가
    room_number = models.CharField(max_length=20, default="101호") # 추가
    is_read = models.BooleanField(default=False)

    def __str__(self):
        return f"[{self.timestamp}] {self.message}"
