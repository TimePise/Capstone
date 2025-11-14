from django.db import models


class Member(models.Model):
    name = models.CharField(
        max_length=50,
        db_column='name',
        help_text="담당자 이름"
    )
    member_id = models.CharField(
        max_length=50,
        primary_key=True,
        unique=True,
        db_column='member_id',
        help_text="병동 계정 ID (예: wardA101)"
    )
    passwd = models.CharField(
        max_length=128,
        db_column='passwd',
        help_text="로그인 비밀번호"
    )
    ward_name = models.CharField(
        max_length=100,
        db_column='ward_name',
        help_text="병동 이름 (예: A동중등, 응급병동 등)"
    )
    phone = models.CharField(
        max_length=20,
        db_column='phone',
        null=True,
        blank=True,
        help_text="전화번호"
    )
    birth_date = models.DateField(
        db_column='birth_date',
        null=True,
        blank=True,
        help_text="생년월일"
    )

    usage_flag = models.CharField(
        max_length=1,
        choices=[('Y', '사용'), ('N', '미사용')],
        default='Y',
        db_column='usage_flag',
        help_text="사용 여부 플래그"
    )
    reg_date = models.DateTimeField(
        auto_now_add=True,
        db_column='reg_date',
        help_text="계정 등록 일자"
    )
    update_date = models.DateTimeField(
        auto_now=True,
        db_column='update_date',
        help_text="계정 수정 일자"
    )

    class Meta:
        db_table = 'member'

    def __str__(self):
        return f"[{self.member_id}] {self.ward_name}"

class UserLog(models.Model):
    id = models.AutoField(
        primary_key=True,
        db_column='id'
    )
    member = models.ForeignKey(
        Member,
        on_delete=models.CASCADE,
        db_column='member_id',
        help_text="로그인한 계정 ID"
    )
    action = models.CharField(
        max_length=20,
        choices=[
            ("signup", "회원가입"),
            ("login", "로그인"),
            ("logout", "로그아웃")
        ],
        db_column='action',
        help_text="로그 유형"
    )
    reg_date = models.DateTimeField(
        auto_now_add=True,
        db_column='reg_date',
        help_text="로그 시간"
    )

    class Meta:
        db_table = 'user_log'

    def __str__(self):
        return f"{self.member.member_id} - {self.action} ({self.reg_date.strftime('%Y-%m-%d %H:%M:%S')})"
class FallRecord(models.Model):
    record_id = models.AutoField(
        primary_key=True,
        db_column='record_id',
        help_text="낙상 기록 고유 ID"
    )
    member = models.ForeignKey(
        Member,
        on_delete=models.CASCADE,
        db_column='member_id',
        help_text="어느 병동에서 발생한 낙상인지 연결"
    )
    name = models.CharField(
        max_length=100,
        db_column='name',
        help_text="환자 이름"
    )
    age = models.PositiveIntegerField(
        db_column='age',
        help_text="환자 나이"
    )
    room_number = models.CharField(
        max_length=20,
        db_column='room_number',
        help_text="호실 (예: 101A)"
    )
    fall_date = models.DateTimeField(
        db_column='fall_date',
        help_text="낙상 발생 일시"
    )
    fall_level = models.CharField(
        max_length=50,
        db_column='fall_level',
        choices=[
            ('경미', '경미'),
            ('중간', '중간'),
            ('심각', '심각')
        ],
        help_text="낙상 단계"
    )
    fall_area = models.CharField(
        max_length=100,
        db_column='fall_area',
        help_text="낙상 부위"
    )
    note = models.TextField(
        db_column='note',
        blank=True,
        null=True,
        help_text="특이사항 (간호기록, 확인 등)"
    )

    class Meta:
        db_table = 'fall_record'
        ordering = ['-fall_date']

    def __str__(self):
        return f"{self.name} ({self.age}세) - {self.fall_date.strftime('%Y-%m-%d %H:%M')}"

# FallRecord, UserLog는 그대로 두고 유지 ✅
