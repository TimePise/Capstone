from django.urls import path, include  # include 추가 ✅
from . import views

urlpatterns = [
    path('', views.index, name='member_index'),

    # 회원가입, 로그인, 로그아웃,계정탈퇴,아이디 비밀번호 찾기, 인증번호
    path('reg/', views.member_reg, name='member_reg'),
    path('login/', views.member_login, name='member_login'),
    path('logout/', views.member_logout, name='member_logout'),
    path('delete/', views.member_delete, name='member_delete'),
    path('find/id/', views.find_id, name='find_id'),
    path('verify_id/', views.verify_id, name='verify_id'),
    path('find/password/', views.find_password, name='find_password'),
    path('send-code/', views.send_verification_code, name='send_verification_code'),  # 인증번호 전송

    # 마이페이지
    path('mypage/', views.mypage, name='mypage'),

    # 낙상 감지 시스템 (페이지)
    path('fall_prevention/', views.fall_prevention, name='fall_prevention'),

    # ✅ fall 앱 내부 기능 추가 (스트리밍 등)
    path('fall_prevention/', include('fall.urls')),

    # 낙상 기록
    path('fall/add/', views.fall_record_add, name='fall_record_add'),
    path('fall/list/', views.fall_record_list, name='fall_record_list'),
    path('fall/alert/list/', views.fall_alert_list, name='fall_alert_list'),  # ✅ 진짜 알림 목록

   
]
