from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.member_reg, name='member_reg'),
    path('login/', views.member_login, name='member_login'),
    path('logout/', views.member_logout, name='member_logout'),
    
    # 낙상방지 시스템 페이지
    path('fall_prevention/', views.fall_prevention, name='fall_prevention'),

    # ✅ 실시간 포즈 추정 영상 스트리밍
    path('pose_feed/', views.pose_estimation_feed, name='pose_feed'),  

    # ✅ 프라이버시 모드 토글 URL 추가
    path('toggle_privacy/', views.toggle_privacy_mode, name='toggle_privacy'),
]
