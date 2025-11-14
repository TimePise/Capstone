from django.apps import AppConfig
import os

class FallConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fall'

    def ready(self):  # ✅ 클래스 안쪽으로 들여쓰기
        if os.environ.get("RUN_MAIN") != "true":
            return
        print("🌀카메라 작동")
        try:
            from .views import start_pose_thread_once
            start_pose_thread_once()
        except Exception as e:
            print("❌ 감지 쓰레드 실행 실패:", e)
