"""
Enhanced URL Configuration
Place this file at: diet_project/urls.py
"""

from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static
from diet_app import views

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # Home and Main Features
    path('', views.home, name='home'),
    path('calculate/', views.calculate_diet, name='calculate_diet'),
    
    # User Authentication
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
    
    # User Dashboard and Profile
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('history/', views.history, name='history'),
    path('add-weight/', views.add_weight, name='add_weight'),
    
    # ML Model Analytics
    path('model-analytics/', views.model_analytics, name='model_analytics'),
    
    # API Endpoints for Charts
    path('api/weight-chart/', views.weight_chart_data, name='weight_chart_data'),
    path('api/bmi-history/', views.bmi_history_data, name='bmi_history_data'),
    path('api/calories-history/', views.calories_history_data, name='calories_history_data'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static('/media/ml_models/', document_root=settings.BASE_DIR / 'ml_models')