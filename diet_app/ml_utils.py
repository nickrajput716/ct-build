"""
Enhanced ML Utilities with Model Loading and Analytics
Place this file at: diet_app/ml_utils.py
"""

import os
import pickle
import numpy as np
from django.conf import settings

# ML Model paths
ML_MODELS_DIR = os.path.join(settings.BASE_DIR, 'ml_models')
VISUALIZATIONS_DIR = os.path.join(ML_MODELS_DIR, 'visualizations')


class MLModelManager:
    """Manager class for ML models with caching"""
    
    def __init__(self):
        self._bmi_classifier = None
        self._tdee_regressor = None
        self._label_encoder_category = None
        self._label_encoder_gender = None
    
    @property
    def bmi_classifier(self):
        if self._bmi_classifier is None:
            path = os.path.join(ML_MODELS_DIR, 'bmi_classifier.pkl')
            with open(path, 'rb') as f:
                self._bmi_classifier = pickle.load(f)
        return self._bmi_classifier
    
    @property
    def tdee_regressor(self):
        if self._tdee_regressor is None:
            path = os.path.join(ML_MODELS_DIR, 'tdee_regressor.pkl')
            with open(path, 'rb') as f:
                self._tdee_regressor = pickle.load(f)
        return self._tdee_regressor
    
    @property
    def label_encoder_category(self):
        if self._label_encoder_category is None:
            path = os.path.join(ML_MODELS_DIR, 'label_encoder_category.pkl')
            with open(path, 'rb') as f:
                self._label_encoder_category = pickle.load(f)
        return self._label_encoder_category
    
    @property
    def label_encoder_gender(self):
        if self._label_encoder_gender is None:
            path = os.path.join(ML_MODELS_DIR, 'label_encoder_gender.pkl')
            with open(path, 'rb') as f:
                self._label_encoder_gender = pickle.load(f)
        return self._label_encoder_gender


# Global instance
ml_manager = MLModelManager()


def calculate_bmi(weight, height):
    """
    Calculate BMI
    Args:
        weight: Weight in kg
        height: Height in cm
    Returns:
        BMI value (float)
    """
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)


def predict_bmi_category(gender, height, weight):
    """
    Predict BMI category using ML model
    Args:
        gender: 'male' or 'female'
        height: Height in cm
        weight: Weight in kg
    Returns:
        category (str): BMI category
        confidence (float): Prediction confidence
    """
    try:
        # Encode gender
        gender_encoded = ml_manager.label_encoder_gender.transform([gender])[0]
        
        # Prepare features
        features = np.array([[gender_encoded, height, weight]])
        
        # Predict
        prediction = ml_manager.bmi_classifier.predict(features)[0]
        category = ml_manager.label_encoder_category.inverse_transform([prediction])[0]
        
        # Get confidence
        probabilities = ml_manager.bmi_classifier.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
        
        return category, confidence
    
    except Exception as e:
        # Fallback to rule-based
        bmi = calculate_bmi(weight, height)
        if bmi < 18.5:
            category = 'Underweight'
        elif 18.5 <= bmi < 25:
            category = 'Normal'
        elif 25 <= bmi < 30:
            category = 'Overweight'
        else:
            category = 'Obese'
        return category, 0.95


def calculate_tdee(age, gender, weight, height, activity_level):
    """
    Calculate TDEE using ML model
    Args:
        age: Age in years
        gender: 'male' or 'female'
        weight: Weight in kg
        height: Height in cm
        activity_level: Activity multiplier
    Returns:
        tdee (int): Predicted TDEE in calories
    """
    try:
        # Map activity level to multiplier
        activity_map = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'veryActive': 1.725,
            'extraActive': 1.9
        }
        activity_multiplier = activity_map.get(activity_level, 1.55)
        
        # Encode gender
        gender_encoded = 1 if gender.lower() == 'male' else 0
        
        # Prepare features
        features = np.array([[age, gender_encoded, weight, height, activity_multiplier]])
        
        # Predict using ML model
        tdee = ml_manager.tdee_regressor.predict(features)[0]
        
        return int(round(tdee))
    
    except Exception as e:
        # Fallback to Mifflin-St Jeor equation
        if gender.lower() == 'male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        
        activity_map = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'veryActive': 1.725,
            'extraActive': 1.9
        }
        activity_multiplier = activity_map.get(activity_level, 1.55)
        tdee = bmr * activity_multiplier
        
        return int(round(tdee))


def adjust_calories_for_goal(tdee, goal):
    """
    Adjust TDEE based on user's goal
    Args:
        tdee: Total Daily Energy Expenditure
        goal: 'lose', 'maintain', or 'gain'
    Returns:
        recommended_calories (int)
    """
    if goal == 'lose':
        return int(tdee - 500)  # 500 cal deficit for ~0.5kg/week loss
    elif goal == 'gain':
        return int(tdee + 500)  # 500 cal surplus for ~0.5kg/week gain
    else:
        return tdee


def get_diet_plan(bmi_category, diet_type, recommended_calories):
    """
    Generate personalized diet plan based on BMI category and diet preference
    """
    
    # Diet plans database
    diet_plans = {
        'Underweight': {
            'veg': {
                'title': 'High-Calorie Vegetarian Plan for Weight Gain',
                'meals': [
                    '🌅 Breakfast: Oatmeal with banana, nuts, honey + Whole milk smoothie (600 cal)',
                    '🍽️ Mid-Morning: Paneer sandwich with cheese + Fruit juice (400 cal)',
                    '🍛 Lunch: Rice, dal, paneer curry, mixed vegetables, curd (800 cal)',
                    '☕ Evening: Protein smoothie with peanut butter, banana (500 cal)',
                    '🌙 Dinner: Chapati, mixed dal, vegetable curry, salad (700 cal)',
                    '🌜 Before Bed: Warm milk with almonds and dates (300 cal)'
                ],
                'tips': [
                    'Eat every 2-3 hours to increase calorie intake',
                    'Include healthy fats: nuts, seeds, avocado, olive oil',
                    'Drink protein shakes between meals',
                    'Focus on nutrient-dense foods',
                    'Consider weight training to build muscle mass'
                ]
            },
            'nonveg': {
                'title': 'High-Protein Non-Veg Plan for Healthy Weight Gain',
                'meals': [
                    '🌅 Breakfast: Eggs (3), whole wheat toast, avocado + Protein shake (650 cal)',
                    '🍽️ Mid-Morning: Chicken sandwich with cheese + Banana (450 cal)',
                    '🍛 Lunch: Rice, chicken curry, dal, vegetables, salad (900 cal)',
                    '☕ Evening: Tuna sandwich + Nuts (500 cal)',
                    '🌙 Dinner: Chapati, fish/chicken, vegetables, dal (750 cal)',
                    '🌜 Before Bed: Greek yogurt with honey and nuts (350 cal)'
                ],
                'tips': [
                    'Include lean protein in every meal',
                    'Eat 5-6 meals per day',
                    'Stay hydrated with water and juices',
                    'Combine strength training with diet',
                    'Get adequate sleep for muscle recovery'
                ]
            },
            'vegan': {
                'title': 'Plant-Based High-Calorie Plan for Weight Gain',
                'meals': [
                    '🌅 Breakfast: Oatmeal with almond butter, chia seeds, berries (600 cal)',
                    '🍽️ Mid-Morning: Hummus with whole grain crackers + Smoothie (450 cal)',
                    '🍛 Lunch: Quinoa bowl, chickpeas, tahini, vegetables (850 cal)',
                    '☕ Evening: Peanut butter banana smoothie + Trail mix (500 cal)',
                    '🌙 Dinner: Brown rice, tofu curry, lentil dal, vegetables (700 cal)',
                    '🌜 Before Bed: Soy milk with dates and walnuts (300 cal)'
                ],
                'tips': [
                    'Focus on calorie-dense plant foods',
                    'Include protein from legumes, tofu, tempeh',
                    'Use healthy oils and nut butters',
                    'Consider vegan protein supplements',
                    'Track calories to ensure surplus'
                ]
            }
        },
        'Normal': {
            'veg': {
                'title': 'Balanced Vegetarian Maintenance Plan',
                'meals': [
                    '🌅 Breakfast: Poha/Upma with vegetables + Green tea (400 cal)',
                    '🍽️ Mid-Morning: Fresh fruits + Handful of nuts (200 cal)',
                    '🍛 Lunch: Chapati (2), dal, vegetable curry, salad, curd (600 cal)',
                    '☕ Evening: Sprouts chaat + Green tea (250 cal)',
                    '🌙 Dinner: Rice/Chapati, dal, vegetables, soup (550 cal)'
                ],
                'tips': [
                    'Maintain balanced macronutrients',
                    'Stay active with regular exercise',
                    'Drink 8-10 glasses of water daily',
                    'Include variety in your diet',
                    'Practice portion control'
                ]
            },
            'nonveg': {
                'title': 'Balanced Non-Veg Maintenance Plan',
                'meals': [
                    '🌅 Breakfast: Eggs (2), whole wheat toast + Coffee (400 cal)',
                    '🍽️ Mid-Morning: Greek yogurt + Fruits (200 cal)',
                    '🍛 Lunch: Chapati (2), chicken/fish, vegetables, salad (650 cal)',
                    '☕ Evening: Boiled eggs or nuts + Tea (200 cal)',
                    '🌙 Dinner: Rice/Chapati, grilled chicken/fish, soup (550 cal)'
                ],
                'tips': [
                    'Choose lean protein sources',
                    'Eat fish 2-3 times per week for omega-3',
                    'Balance carbs, proteins, and healthy fats',
                    'Stay consistent with meal timing',
                    'Combine with 30 min daily exercise'
                ]
            },
            'vegan': {
                'title': 'Balanced Plant-Based Maintenance Plan',
                'meals': [
                    '🌅 Breakfast: Smoothie bowl with fruits, seeds, granola (400 cal)',
                    '🍽️ Mid-Morning: Apple with almond butter (200 cal)',
                    '🍛 Lunch: Quinoa, chickpeas, roasted vegetables, tahini (600 cal)',
                    '☕ Evening: Hummus with veggie sticks (250 cal)',
                    '🌙 Dinner: Brown rice, lentil curry, vegetables (550 cal)'
                ],
                'tips': [
                    'Ensure adequate B12 supplementation',
                    'Combine different protein sources',
                    'Include iron-rich foods with vitamin C',
                    'Eat rainbow of colorful vegetables',
                    'Monitor nutrient intake regularly'
                ]
            }
        },
        'Overweight': {
            'veg': {
                'title': 'Vegetarian Weight Loss Plan',
                'meals': [
                    '🌅 Breakfast: Vegetable poha + Green tea (350 cal)',
                    '🍽️ Mid-Morning: Fruits (100 cal)',
                    '🍛 Lunch: Chapati (1), dal, vegetables, salad, buttermilk (450 cal)',
                    '☕ Evening: Green tea + Roasted chana (150 cal)',
                    '🌙 Dinner: Vegetable soup + Salad + Paneer (400 cal)'
                ],
                'tips': [
                    'Create 500 calorie daily deficit',
                    'Exercise 45-60 minutes daily',
                    'Avoid fried and processed foods',
                    'Drink water before meals',
                    'Get 7-8 hours sleep'
                ]
            },
            'nonveg': {
                'title': 'High-Protein Weight Loss Plan',
                'meals': [
                    '🌅 Breakfast: Egg white omelette + Green tea (300 cal)',
                    '🍽️ Mid-Morning: Fruits (100 cal)',
                    '🍛 Lunch: Grilled chicken/fish + Vegetables + Salad (500 cal)',
                    '☕ Evening: Green tea + Boiled egg (100 cal)',
                    '🌙 Dinner: Grilled fish/chicken + Soup + Salad (450 cal)'
                ],
                'tips': [
                    'Prioritize lean protein sources',
                    'Avoid late night eating',
                    'Do cardio and strength training',
                    'Stay consistent with deficit',
                    'Track your progress weekly'
                ]
            },
            'vegan': {
                'title': 'Plant-Based Weight Loss Plan',
                'meals': [
                    '🌅 Breakfast: Green smoothie + Chia seeds (300 cal)',
                    '🍽️ Mid-Morning: Berries (100 cal)',
                    '🍛 Lunch: Quinoa salad with legumes + Vegetables (450 cal)',
                    '☕ Evening: Herbal tea + Handful almonds (150 cal)',
                    '🌙 Dinner: Vegetable soup + Tofu + Mixed greens (400 cal)'
                ],
                'tips': [
                    'Focus on whole, unprocessed foods',
                    'Include fiber-rich foods for satiety',
                    'Avoid high-calorie plant fats initially',
                    'Stay active throughout the day',
                    'Practice mindful eating'
                ]
            }
        },
        'Obese': {
            'veg': {
                'title': 'Intensive Vegetarian Weight Loss Plan',
                'meals': [
                    '🌅 Breakfast: Vegetable oats + Green tea (300 cal)',
                    '🍽️ Mid-Morning: Cucumber + Carrot sticks (50 cal)',
                    '🍛 Lunch: Chapati (1), dal, vegetables, large salad (400 cal)',
                    '☕ Evening: Green tea + Sprouts (100 cal)',
                    '🌙 Dinner: Clear soup + Grilled vegetables + Salad (350 cal)'
                ],
                'tips': [
                    'Consult healthcare provider before starting',
                    'Create 750-1000 calorie deficit',
                    'Exercise daily (start with walking)',
                    'Drink 10-12 glasses water',
                    'Consider professional guidance',
                    'Focus on long-term lifestyle changes'
                ]
            },
            'nonveg': {
                'title': 'Intensive Protein-Rich Weight Loss Plan',
                'meals': [
                    '🌅 Breakfast: Egg whites (4) + Vegetables + Tea (250 cal)',
                    '🍽️ Mid-Morning: Apple (80 cal)',
                    '🍛 Lunch: Grilled chicken breast + Vegetables + Salad (450 cal)',
                    '☕ Evening: Green tea + Boiled egg whites (80 cal)',
                    '🌙 Dinner: Grilled fish + Clear soup + Large salad (400 cal)'
                ],
                'tips': [
                    'Medical supervision recommended',
                    'High protein to preserve muscle',
                    'Gradual exercise progression',
                    'Address emotional eating',
                    'Join support groups',
                    'Celebrate small victories'
                ]
            },
            'vegan': {
                'title': 'Intensive Plant-Based Weight Loss Plan',
                'meals': [
                    '🌅 Breakfast: Green smoothie + Flax seeds (280 cal)',
                    '🍽️ Mid-Morning: Celery sticks (40 cal)',
                    '🍛 Lunch: Lentil soup + Large vegetable salad (400 cal)',
                    '☕ Evening: Herbal tea + Cherry tomatoes (60 cal)',
                    '🌙 Dinner: Steamed tofu + Vegetables + Soup (380 cal)'
                ],
                'tips': [
                    'Seek professional nutritionist guidance',
                    'Focus on nutrient density',
                    'Track all food intake',
                    'Build sustainable habits',
                    'Address underlying health issues',
                    'Be patient with progress'
                ]
            }
        }
    }
    
    return diet_plans.get(bmi_category, {}).get(diet_type, diet_plans['Normal']['veg'])


def get_visualization_path(viz_name):
    """Get path to visualization image"""
    return os.path.join(VISUALIZATIONS_DIR, viz_name)


def get_model_metrics():
    """Get stored model performance metrics"""
    metrics = {
        'bmi_classifier': {
            'confusion_matrix': 'bmi_confusion_matrix.png',
            'learning_curve': 'bmi_learning_curve.png',
            'feature_importance': 'bmi_feature_importance.png'
        },
        'tdee_regressor': {
            'regression_plot': 'tdee_regression_plot.png',
            'learning_curve': 'tdee_learning_curve.png',
            'feature_importance': 'tdee_feature_importance.png'
        }
    }
    return metrics