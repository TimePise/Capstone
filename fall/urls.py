from django.urls import path
from . import views

urlpatterns = [
    path('pose_feed/', views.pose_estimation_feed, name='pose_estimation_feed'),
    path('privacy/', views.toggle_privacy_mode, name='toggle_privacy_mode'),
    path('fall_status/', views.fall_status, name='fall_status'),  # 💡 여기 오타 수정됨!
]
