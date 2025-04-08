# config/asgi.py

import os
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.core.asgi import get_asgi_application
import member.routing  # ✅ 여기에 WebSocket URL 라우팅이 있어야 함

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            member.routing.websocket_urlpatterns
        )
    ),
})
