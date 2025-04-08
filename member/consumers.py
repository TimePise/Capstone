# member/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class FallAlertConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # 그룹 이름은 고정 "fall_alerts"
        await self.channel_layer.group_add("fall_alerts", self.channel_name)
        await self.accept()
        print("✅ WebSocket 연결됨")

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("fall_alerts", self.channel_name)
        print("❌ WebSocket 연결 해제됨")

    async def receive(self, text_data):
        # 클라이언트에서 메시지 받을 경우 (안쓸 수도 있음)
        data = json.loads(text_data)
        print("📨 받은 메시지:", data)

    async def send_fall_alert(self, event):
        message = event["message"]
        # 클라이언트에게 보낼 실제 메시지
        await self.send(text_data=json.dumps({
            "message": message
        }))