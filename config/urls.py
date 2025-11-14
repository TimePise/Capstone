from django.contrib import admin
from django.urls import path
from django.conf.urls import include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('member/', include('member.urls')),
    path('fall/', include('fall.urls')),  # ✅ 이 줄이 반드시 필요!!
]
