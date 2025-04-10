from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse
from .models import Member, UserLog, FallRecord
import cv2
import mediapipe as mp
import random
from datetime import datetime
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 전역 프라이버시 모드 상태
privacy_mode = False

# 인증번호 저장 (임시 메모리, 실제로는 세션이나 DB 사용 권장)
verification_store = {}
def generate_code():
    return str(random.randint(100000, 999999))

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

# ✅ 계정 삭제 (탈퇴)
def member_delete(request):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")

    member = get_object_or_404(Member, member_id=member_id)

    if request.method == "POST":
        # 로그 기록 남기기
        UserLog.objects.create(member=member, action="logout")

        # 회원 삭제
        member.delete()
        request.session.flush()  # 세션 클리어
        return redirect("member_login")

    return render(request, "member/member_delete.html", {"member": member})

# ✅ 아이디,비밀번호 찾기 및 인증번호

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


def send_verification_code(request):
    phone = request.POST.get('phone')
    code = generate_code()
    verification_store[phone] = code
    print(f"📨 {phone} 번호로 인증번호 전송됨: {code}")  # 실제로는 문자 API로 전송
    return JsonResponse({'status': '인증번호가 전송되었습니다.'})

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
        else:
            return JsonResponse({'error': '해당 정보와 일치하는 계정을 찾을 수 없습니다.'}, status=404)

    return render(request, 'member/find_id.html')


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
        else:
            return JsonResponse({'error': '일치하는 정보가 없습니다.'}, status=404)

    return render(request, 'member/find_password.html')

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

def fall_record_add(request):
    member_id = request.session.get("m_id")
    if not member_id:
        return redirect("member_login")

    member = get_object_or_404(Member, member_id=member_id)

    if request.method == "POST":
        record = FallRecord.objects.create(
            member=member,
            name=request.POST["name"],
            age=request.POST["age"],
            room_number=request.POST["room_number"],
            fall_date=request.POST["fall_date"],
            fall_level=request.POST["fall_level"],
            fall_area=request.POST["fall_area"],
            note=request.POST.get("note", "")
        )

        # WebSocket으로 알림 전송
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            "fall_alert_group",
            {
                "type": "send_fall_alert",
                "message": f"{record.name} 환자의 낙상 발생!"
            }
        )

        return redirect("fall_record_list")
    
    ###수동으로 낙상 알림을 전송하는 테스트용 뷰
def test_fall_alert(request):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "fall_alerts",
        {
            "type": "send_fall_alert",
            "message": "⚠️ 테스트 낙상 알림이 발생했습니다!"
        }
    )
    return JsonResponse({"status": "알림 전송 완료!"})

# ✅ 실시간 영상 스트리밍
# 기존의 generate_frames 함수
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
        draw_frame = frame.copy()

        # ✅ 랜드마크 그리기
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                draw_frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            )

        # ✅ 보호모드 오버레이 및 텍스트 표시 (Pillow 사용)
        if privacy_mode:
            overlay = np.zeros_like(draw_frame)

            # PIL 이미지로 변환
            pil_img = Image.fromarray(overlay)
            draw = ImageDraw.Draw(pil_img)

            try:
                # 윈도우 시스템 기본 한글 폰트 (영어만 쓸 거여도 안전하게 로드)
                font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 32)
            except:
                font = ImageFont.load_default()

            draw.text((50, 50), "Privacy Mode ON", font=font, fill=(255, 255, 255))

            # PIL → numpy(OpenCV) 변환
            overlay = np.array(pil_img)

            # 프레임 위에 오버레이 덮기
            draw_frame = cv2.addWeighted(draw_frame, 0.2, overlay, 0.8, 0)

        # 프레임을 JPEG 인코딩
        _, buffer = cv2.imencode('.jpg', draw_frame)
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
