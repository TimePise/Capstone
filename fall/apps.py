# fall/apps.py
from django.apps import AppConfig

class FallConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fall'

    def ready(self):
        from .views import start_pose_thread_once
        start_pose_thread_once()  # ✅ 반드시 이 줄이 있어야 함
