from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.http import JsonResponse, HttpResponse
from django.db.models import Count
from django.utils import timezone
from member.models import Member, UserLog, FallRecord
from fall.models import FallAlert
from fall.views import get_privacy_mode_state
import csv
import random
from datetime import datetime, timedelta
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# 인증번호 저장소
verification_store = {}
def generate_code():
    return str(random.randint(100000, 999999))

# ✅ index
def index(request):
    context = {
        "m_id": request.session.get("m_id", ""),
        "m_name": request.session.get("m_name", "")
    }
    return render(request, "member/index.html", context)

# ✅ 회원가입
@csrf_exempt
def member_reg(request):
    if request.method == "GET":
        return render(request, "member/member_reg.html")
    elif request.method == "POST":
        member_id = request.POST.get("member_id")
        passwd = request.POST.get("passwd")
        name = request.POST.get("name")
        ward_name = request.POST.get("ward_name")
        phone = request.POST.get("phone")
        code = request.POST.get("code")

        if verification_store.get(phone) != code:
            return render(request, "member/member_reg.html", {
                "message": "인증번호가 올바르지 않습니다.",
                "member_id": member_id,
                "name": name,
                "ward_name": ward_name,
                "phone": phone,
            })

        if Member.objects.filter(member_id=member_id).exists():
            return render(request, "member/member_reg.html", {
                "message": f"{member_id}는 이미 존재하는 아이디입니다.",
                "member_id": member_id,
                "name": name,
                "ward_name": ward_name,
                "phone": phone,
            })

        member = Member.objects.create(
            member_id=member_id,
            passwd=passwd,
            name=name,
            ward_name=ward_name,
            phone=phone,
        )
        UserLog.objects.create(member=member, action="signup")

        request.session["signup_success"] = True
        return redirect("member_login")

# ✅ 로그인
@csrf_exempt
def member_login(request):
    if request.method == "GET":
        signup_success = request.session.pop("signup_success", False)
        return render(request, "member/login.html", {"signup_success": signup_success})
    elif request.method == "POST":
        member_id = request.POST.get("member_id")
        passwd = request.POST.get("passwd")
        member = Member.objects.filter(member_id=member_id, passwd=passwd).first()
        if member:
            request.session["m_id"] = member.member_id
            request.session["m_name"] = member.name
            UserLog.objects.create(member=member, action="login")
            return redirect("fall_prevention")
        return render(request, "member/login.html", {
            "message": "아이디 또는 비밀번호가 일치하지 않습니다.",
            "signup_success": False,
        })

# ✅ 로그아웃
def member_logout(request):
    member_id = request.session.get("m_id")
    if member_id:
        member = get_object_or_404(Member, member_id=member_id)
        UserLog.objects.create(member=member, action="logout")
    request.session.flush()
    return redirect("member_login")

# ✅ 계정 삭제
def member_delete(request):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")

    member = get_object_or_404(Member, member_id=member_id)

    if request.method == "POST":
        UserLog.objects.create(member=member, action="logout")
        member.delete()
        request.session.flush()
        return redirect("member_login")

    return render(request, "member/member_delete.html", {"member": member})

# ✅ 인증번호 전송
def send_verification_code(request):
    phone = request.POST.get('phone')
    code = generate_code()
    verification_store[phone] = code
    print(f"📨 {phone} → 인증번호 전송됨: {code}")
    return JsonResponse({'status': '인증번호가 전송되었습니다.'})

# ✅ ID 확인
def verify_id(request):
    if request.method == "POST":
        name = request.POST.get("name")
        phone = request.POST.get("phone")
        code = request.POST.get("code")

        if verification_store.get(phone) != code:
            return JsonResponse({'status': '인증번호가 올바르지 않습니다.'}, status=400)

        try:
            member = Member.objects.get(name=name, phone=phone)
            return JsonResponse({'member_id': member.member_id})
        except Member.DoesNotExist:
            return JsonResponse({'status': '일치하는 회원 정보가 없습니다.'}, status=404)

# ✅ ID 찾기
def find_id(request):
    if request.method == "POST":
        name = request.POST.get('name')
        phone = request.POST.get('phone')
        code = request.POST.get('code')

        if verification_store.get(phone) != code:
            return JsonResponse({'error': '인증번호가 올바르지 않습니다.'}, status=400)

        member = Member.objects.filter(name=name, phone=phone).first()
        if member:
            return JsonResponse({'member_id': member.member_id})
        return JsonResponse({'error': '일치하는 계정을 찾을 수 없습니다.'}, status=404)

    return render(request, 'member/find_id.html')

# ✅ 비밀번호 찾기
def find_password(request):
    if request.method == "POST":
        member_id = request.POST.get('member_id')
        name = request.POST.get('name')
        phone = request.POST.get('phone')
        code = request.POST.get('code')

        if verification_store.get(phone) != code:
            return JsonResponse({'error': '인증번호가 올바르지 않습니다.'}, status=400)

        member = Member.objects.filter(member_id=member_id, name=name, phone=phone).first()
        if member:
            return JsonResponse({'passwd': member.passwd})
        return JsonResponse({'error': '일치하는 정보가 없습니다.'}, status=404)

    return render(request, 'member/find_password.html')

# ✅ 마이페이지
def mypage(request):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")

    member = get_object_or_404(Member, member_id=member_id)

    phone = member.phone or "-"
    if phone and phone.isdigit() and len(phone) == 11:
        phone = f"{phone[:3]}-{phone[3:7]}-{phone[7:]}"

    ward_name = f"{member.ward_name}병동" if member.ward_name else "-"

    return render(request, "member/mypage.html", {
        "member": member,
        "formatted_phone": phone,
        "formatted_ward": ward_name,
    })

# ✅ 낙상 감지 페이지
def fall_prevention(request):
    if not request.session.get("m_id"):
        return redirect("member_login")
    return render(request, "member/fall_prevention.html", {
        "m_id": request.session.get("m_id", ""),
        "m_name": request.session.get("m_name", ""),
        "privacy_mode": get_privacy_mode_state(),
    })

@csrf_exempt
def fall_record_add(request):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")
    member = get_object_or_404(Member, member_id=member_id)

    if request.method == "GET":
        return render(request, "member/fall_record_add.html",{"m_id": member.member_id})
    elif request.method == "POST":
        FallRecord.objects.create(
            member=member,
            name=request.POST["name"],
            age=request.POST["age"],
            room_number=request.POST["room_number"],
            fall_date=request.POST["fall_date"],
            fall_level=request.POST["fall_level"],
            fall_area=request.POST["fall_area"],
            note=request.POST.get("note", "")
        )
        return redirect("fall_record_list")


def _filtered_records(request, member):
    records = FallRecord.objects.filter(member=member).order_by("-fall_date")
    filters = {
        "name": request.GET.get("name", "").strip(),
        "level": request.GET.get("level", "").strip(),
        "start_date": request.GET.get("start_date", "").strip(),
        "end_date": request.GET.get("end_date", "").strip(),
    }

    if filters["name"]:
        records = records.filter(name__icontains=filters["name"])
    if filters["level"]:
        records = records.filter(fall_level=filters["level"])
    if filters["start_date"]:
        records = records.filter(fall_date__date__gte=filters["start_date"])
    if filters["end_date"]:
        records = records.filter(fall_date__date__lte=filters["end_date"])

    return records, filters


def _build_record_summary(records):
    level_order = ["심각", "중간", "경미"]
    level_counts = {level: 0 for level in level_order}
    for row in records.values("fall_level").annotate(count=Count("record_id")):
        if row["fall_level"] in level_counts:
            level_counts[row["fall_level"]] = row["count"]

    today = timezone.localdate()
    start_day = today - timedelta(days=6)
    daily_template = {start_day + timedelta(days=i): 0 for i in range(7)}
    trend_rows = (
        records.filter(fall_date__date__gte=start_day)
        .values("fall_date__date")
        .annotate(count=Count("record_id"))
    )
    for row in trend_rows:
        day = row["fall_date__date"]
        if day in daily_template:
            daily_template[day] = row["count"]

    weekly_trend = [
        {"date": day.strftime("%m.%d"), "count": daily_template[day]}
        for day in sorted(daily_template.keys())
    ]

    return {
        "total": records.count(),
        "latest": records.first(),
        "levels": {
            "critical": level_counts.get("심각", 0),
            "warning": level_counts.get("중간", 0),
            "safe": level_counts.get("경미", 0),
        },
        "weekly_trend": weekly_trend,
    }


def fall_record_list(request):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")
    member = get_object_or_404(Member, member_id=member_id)

    records, filters = _filtered_records(request, member)
    summary = _build_record_summary(records)
    recent_alerts = FallAlert.objects.order_by("-timestamp")[:3]

    context = {
        "records": records,
        "m_id": member.member_id,
        "filters": filters,
        "summary": summary,
        "recent_alerts": recent_alerts,
        "export_query": request.GET.urlencode(),
    }
    return render(request, "member/fall_record_list.html", context)

@require_POST
def fall_record_delete(request, record_id):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")
    member = get_object_or_404(Member, member_id=member_id)
    record = get_object_or_404(FallRecord, pk=record_id, member=member)
    record.delete()
    return redirect("fall_record_list")


@require_GET
def fall_record_export(request):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")
    member = get_object_or_404(Member, member_id=member_id)

    records, _ = _filtered_records(request, member)
    timestamp = timezone.now().strftime("%Y%m%d_%H%M")
    filename = f"fall_records_{timestamp}.csv"

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'

    writer = csv.writer(response)
    writer.writerow(["이름", "나이", "병동/호실", "발생일시", "낙상단계", "부위", "특이사항"])
    for record in records:
        writer.writerow([
            record.name,
            record.age,
            record.room_number,
            record.fall_date.strftime("%Y-%m-%d %H:%M"),
            record.fall_level,
            record.fall_area,
            record.note or "",
        ])
    return response


# ✅ 낙상 알림 리스트 
def fall_alert_list(request):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")

    member = get_object_or_404(Member, member_id=member_id)

    filters = {
        "level": request.GET.get("level", "").strip(),
        "status": request.GET.get("status", "").strip(),
    }

    queryset = FallAlert.objects.all().order_by("-timestamp")
    if filters["level"]:
        queryset = queryset.filter(fall_level=filters["level"])
    if filters["status"] == "unread":
        queryset = queryset.filter(is_read=False)
    elif filters["status"] == "read":
        queryset = queryset.filter(is_read=True)

    alerts = list(queryset[:50])

    level_labels = ["고위험", "중위험", "저위험"]
    level_stats = {label: 0 for label in level_labels}
    for row in FallAlert.objects.values("fall_level").annotate(count=Count("id")):
        if row["fall_level"] in level_stats:
            level_stats[row["fall_level"]] = row["count"]

    summary = {
        "total": FallAlert.objects.count(),
        "unread": FallAlert.objects.filter(is_read=False).count(),
        "latest": FallAlert.objects.order_by("-timestamp").first(),
        "levels": level_stats,
    }

    context = {
        "alerts": alerts,
        "m_id": member.member_id,
        "filters": filters,
        "summary": summary,
    }
    return render(request, "member/fall_alert_list.html", context)

@require_POST
def fall_alert_mark(request, alert_id):
    if not request.session.get("m_id"):
        return JsonResponse({"error": "로그인이 필요합니다."}, status=403)

    alert = get_object_or_404(FallAlert, id=alert_id)
    if not alert.is_read:
        alert.is_read = True
        alert.save(update_fields=["is_read"])
    return JsonResponse({"status": "ok"})

