from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='member_index'),

    # 회원가입, 로그인, 로그아웃
    path('reg/', views.member_reg, name='member_reg'),
    path('login/', views.member_login, name='member_login'),
    path('logout/', views.member_logout, name='member_logout'),
    # 마이페이지
    path('mypage/', views.mypage, name='mypage'),
    # 낙상 감지 시스템
    path('fall_prevention/', views.fall_prevention, name='fall_prevention'),
    path('pose_feed/', views.pose_estimation_feed, name='pose_estimation_feed'),
    path('privacy/', views.toggle_privacy_mode, name='toggle_privacy_mode'),
    # 낙상 기록
    path('fall/add/', views.fall_record_add, name='fall_record_add'),
    path('fall/list/', views.fall_record_list, name='fall_record_list'),
]
