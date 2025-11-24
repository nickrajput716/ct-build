from django.contrib import admin
from .models import UserProfile, DietRecommendation, WeightLog


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'age', 'gender', 'weight', 'height', 'diet_type', 'goal']
    list_filter = ['gender', 'diet_type', 'goal', 'activity_level']
    search_fields = ['user__username', 'user__email']


@admin.register(DietRecommendation)
class DietRecommendationAdmin(admin.ModelAdmin):
    list_display = ['user', 'bmi_category', 'bmi', 'recommended_calories', 'created_at']
    list_filter = ['bmi_category', 'diet_type', 'created_at']
    search_fields = ['user__username']
    date_hierarchy = 'created_at'


@admin.register(WeightLog)
class WeightLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'weight', 'date']
    list_filter = ['date']
    search_fields = ['user__username']
    date_hierarchy = 'date'