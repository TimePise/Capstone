from django.contrib import admin
from .models import Member, UserLog, FallRecord

# Member 모델 등록
@admin.register(Member)
class MemberAdmin(admin.ModelAdmin):
    list_display = ['member_id', 'ward_name', 'usage_flag', 'reg_date']

@admin.register(UserLog)
class UserLogAdmin(admin.ModelAdmin):
    list_display = ['member', 'action', 'reg_date']

@admin.register(FallRecord)
class FallRecordAdmin(admin.ModelAdmin):
    list_display = ['name', 'age', 'fall_date', 'fall_level']
