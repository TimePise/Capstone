# fall/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('pose_feed/', views.pose_estimation_feed, name='pose_estimation_feed'),
    path('reset_alert/', views.reset_alert_lock, name='reset_alert_lock'),
    path('toggle_privacy/', views.toggle_privacy_mode, name='toggle_privacy_mode'),
    path('fall_status/', views.fall_status, name='fall_status'),
    # ✅ SSE 알림 스트림 추가
    path('sse/fall_alert/', views.fall_alert_stream, name='fall_alert_stream'),
]
