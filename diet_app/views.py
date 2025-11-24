"""
Enhanced Views with Model Analytics Dashboard
Place this file at: diet_app/views.py
"""

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib import messages
from django.http import JsonResponse
from .forms import UserRegisterForm, UserProfileForm, WeightLogForm
from .models import UserProfile, DietRecommendation, WeightLog
from .ml_utils import (
    calculate_bmi, predict_bmi_category, calculate_tdee,
    adjust_calories_for_goal, get_diet_plan, get_model_metrics,
    get_visualization_path
)
import os
from datetime import datetime, timedelta


def home(request):
    """Home page"""
    return render(request, 'diet_app/home.html')


def calculate_diet(request):
    """Calculate diet recommendation"""
    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.POST.get('age'))
            gender = request.POST.get('gender')
            height = float(request.POST.get('height'))
            weight = float(request.POST.get('weight'))
            activity_level = request.POST.get('activity_level')
            goal = request.POST.get('goal')
            diet_type = request.POST.get('diet_type')
            
            # Calculate BMI
            bmi = calculate_bmi(weight, height)
            
            # Predict BMI category using ML model
            bmi_category, confidence = predict_bmi_category(gender, height, weight)
            
            # Calculate TDEE using ML model
            tdee = calculate_tdee(age, gender, weight, height, activity_level)
            
            # Adjust calories based on goal
            recommended_calories = adjust_calories_for_goal(tdee, goal)
            
            # Get personalized diet plan
            diet_plan = get_diet_plan(bmi_category, diet_type, recommended_calories)
            
            # Prepare result
            result = {
                'bmi': bmi,
                'category': bmi_category,
                'confidence': round(confidence * 100, 1),
                'tdee': tdee,
                'recommended_calories': recommended_calories,
                'diet_type': diet_type,
                'goal': goal,
                'diet_plan': diet_plan
            }
            
            # Save to database if user is authenticated
            if request.user.is_authenticated:
                # Update or create user profile
                profile, created = UserProfile.objects.get_or_create(user=request.user)
                profile.age = age
                profile.gender = gender
                profile.height = height
                profile.weight = weight
                profile.activity_level = activity_level
                profile.goal = goal
                profile.diet_type = diet_type
                profile.save()
                
                # Save recommendation
                DietRecommendation.objects.create(
                    user=request.user,
                    bmi=bmi,
                    bmi_category=bmi_category,
                    tdee=tdee,
                    recommended_calories=recommended_calories,
                    activity_level=activity_level,
                    goal=goal,
                    diet_type=diet_type
                )
                
                messages.success(request, 'Diet plan calculated and saved successfully!')
            
            return render(request, 'diet_app/result.html', {'result': result})
        
        except Exception as e:
            messages.error(request, f'Error calculating diet: {str(e)}')
            return redirect('calculate_diet')
    
    return render(request, 'diet_app/calculate.html')


@login_required
def dashboard(request):
    """User dashboard with analytics"""
    try:
        profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        messages.warning(request, 'Please complete your profile first.')
        return redirect('profile')
    
    # Get latest recommendation
    latest_recommendation = DietRecommendation.objects.filter(
        user=request.user
    ).order_by('-created_at').first()
    
    # Get recommendation history
    recommendations = DietRecommendation.objects.filter(
        user=request.user
    ).order_by('-created_at')[:5]
    
    # Get weight logs
    weight_logs = WeightLog.objects.filter(
        user=request.user
    ).order_by('-date')[:10]
    
    # Calculate progress
    progress_data = None
    if weight_logs.count() >= 2:
        latest_weight = weight_logs.first().weight
        initial_weight = weight_logs.last().weight
        weight_change = latest_weight - initial_weight
        
        progress_data = {
            'initial_weight': initial_weight,
            'current_weight': latest_weight,
            'weight_change': weight_change,
            'change_percentage': round((weight_change / initial_weight) * 100, 1)
        }
    
    context = {
        'profile': profile,
        'latest_recommendation': latest_recommendation,
        'recommendations': recommendations,
        'weight_logs': weight_logs,
        'progress_data': progress_data
    }
    
    return render(request, 'diet_app/dashboard.html', context)


@login_required
def profile(request):
    """User profile page"""
    try:
        user_profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        user_profile = None
    
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=user_profile)
        if form.is_valid():
            profile = form.save(commit=False)
            profile.user = request.user
            profile.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('dashboard')
    else:
        form = UserProfileForm(instance=user_profile)
    
    return render(request, 'diet_app/profile.html', {'form': form})


@login_required
def add_weight(request):
    """Add weight log entry"""
    if request.method == 'POST':
        form = WeightLogForm(request.POST)
        if form.is_valid():
            weight_log = form.save(commit=False)
            weight_log.user = request.user
            weight_log.save()
            
            # Update profile weight
            profile = UserProfile.objects.get(user=request.user)
            profile.weight = weight_log.weight
            profile.save()
            
            messages.success(request, 'Weight logged successfully!')
            return redirect('dashboard')
    else:
        form = WeightLogForm()
    
    return render(request, 'diet_app/add_weight.html', {'form': form})


@login_required
def history(request):
    """View recommendation history"""
    recommendations = DietRecommendation.objects.filter(
        user=request.user
    ).order_by('-created_at')
    
    return render(request, 'diet_app/history.html', {
        'recommendations': recommendations
    })


@login_required
def model_analytics(request):
    """View ML model analytics and visualizations"""
    metrics = get_model_metrics()
    
    # Check if visualization files exist
    visualizations = {}
    for model_name, viz_files in metrics.items():
        visualizations[model_name] = {}
        for viz_type, filename in viz_files.items():
            viz_path = get_visualization_path(filename)
            if os.path.exists(viz_path):
                # Return URL path for template
                visualizations[model_name][viz_type] = f'/media/ml_models/visualizations/{filename}'
            else:
                visualizations[model_name][viz_type] = None
    
    context = {
        'visualizations': visualizations,
        'model_info': {
            'bmi_classifier': {
                'name': 'BMI Classification Model',
                'type': 'Random Forest Classifier',
                'features': ['Gender', 'Height (cm)', 'Weight (kg)'],
                'classes': ['Underweight', 'Normal', 'Overweight', 'Obese'],
                'description': 'Predicts BMI category based on physical attributes using ensemble learning.'
            },
            'tdee_regressor': {
                'name': 'TDEE Prediction Model',
                'type': 'Random Forest Regressor',
                'features': ['Age', 'Gender', 'Weight (kg)', 'Height (cm)', 'Activity Level'],
                'target': 'Total Daily Energy Expenditure (calories)',
                'description': 'Estimates daily calorie requirements using Mifflin-St Jeor enhanced with ML.'
            }
        }
    }
    
    return render(request, 'diet_app/model_analytics.html', context)


def register(request):
    """User registration"""
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Account created for {user.username}!')
            return redirect('profile')
    else:
        form = UserRegisterForm()
    
    return render(request, 'registration/register.html', {'form': form})


# API endpoints for charts
@login_required
def weight_chart_data(request):
    """API endpoint for weight progression chart"""
    weight_logs = WeightLog.objects.filter(
        user=request.user
    ).order_by('date')
    
    data = {
        'dates': [log.date.strftime('%Y-%m-%d') for log in weight_logs],
        'weights': [float(log.weight) for log in weight_logs]
    }
    
    return JsonResponse(data)


@login_required
def bmi_history_data(request):
    """API endpoint for BMI history chart"""
    recommendations = DietRecommendation.objects.filter(
        user=request.user
    ).order_by('created_at')
    
    data = {
        'dates': [rec.created_at.strftime('%Y-%m-%d') for rec in recommendations],
        'bmi_values': [float(rec.bmi) for rec in recommendations],
        'categories': [rec.bmi_category for rec in recommendations]
    }
    
    return JsonResponse(data)


@login_required
def calories_history_data(request):
    """API endpoint for calorie recommendation history"""
    recommendations = DietRecommendation.objects.filter(
        user=request.user
    ).order_by('created_at')
    
    data = {
        'dates': [rec.created_at.strftime('%Y-%m-%d') for rec in recommendations],
        'tdee': [rec.tdee for rec in recommendations],
        'recommended': [rec.recommended_calories for rec in recommendations]
    }
    
    return JsonResponse(data)