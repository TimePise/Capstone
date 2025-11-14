from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse
from member.models import Member, UserLog, FallRecord
from fall.models import FallAlert
import cv2
import mediapipe as mp
import random
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 프라이버시 모드 상태
privacy_mode = False

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
        return render(request, "member/login.html")
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
            "message": "아이디 또는 비밀번호가 일치하지 않습니다."
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
        "privacy_mode": privacy_mode,
    })

# ✅ 실시간 영상 스트리밍
def generate_frames():
    global privacy_mode
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if privacy_mode:
            frame[:] = (0, 0, 0)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            )

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

def pose_estimation_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# ✅ 보호모드 전환
def toggle_privacy_mode(request):
    global privacy_mode
    privacy_mode = not privacy_mode
    return JsonResponse({'privacy_mode': privacy_mode})

# ✅ 낙상 기록 등록
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

# ✅ 낙상 기록 리스트
def fall_record_list(request):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")
    member = get_object_or_404(Member, member_id=member_id)
    records = FallRecord.objects.filter(member=member).order_by("-fall_date")
    return render(request, "member/fall_record_list.html", {"records": records,"m_id": member.member_id})

# ✅ 낙상 알림 리스트 
def fall_alert_list(request):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")

    member = get_object_or_404(Member, member_id=member_id)
    alerts = FallAlert.objects.filter().order_by("-timestamp")[:50]  # 최근 알림 50개
    return render(request, "member/fall_alert_list.html", {"alerts": alerts,"m_id": member.member_id})

