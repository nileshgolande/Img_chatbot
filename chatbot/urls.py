from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chat-api/', views.chat_api, name='chat_api'),
]
