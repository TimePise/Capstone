from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse
from .models import Member, UserLog, FallRecord
import cv2
import mediapipe as mp
from datetime import datetime

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 전역 프라이버시 모드 상태
privacy_mode = False

# ✅ index (홈)
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

        if Member.objects.filter(member_id=member_id).exists():
            return render(request, "member/member_reg.html", {
                "message": f"{member_id}는 이미 존재하는 아이디입니다."
            })

        member = Member.objects.create(
            member_id=member_id,
            passwd=passwd,
            name=name,
            ward_name=ward_name,
            phone=phone,
        )

        UserLog.objects.create(member=member, action="signup")
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
        else:
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

# ✅ 마이페이지
def mypage(request):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")
    member = get_object_or_404(Member, member_id=member_id)
    context = {
        "member": member,
        "name": member.name,
        "member_id": member.member_id,
        "phone": member.phone or "정보 없음",
    }
    return render(request, "member/mypage.html", context)

# ✅ 낙상 감지 페이지
def fall_prevention(request):
    return render(request, "member/fall_prevention.html", {
        "m_id": request.session.get("m_id", ""),
        "m_name": request.session.get("m_name", ""),
        "privacy_mode": privacy_mode
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
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3)
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
        return render(request, "member/fall_record_add.html")

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
    return render(request, "member/fall_record_list.html", {"records": records})
