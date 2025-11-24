"""
Enhanced ML Training Script with Model Evaluation, Confusion Matrix, and Learning Curves
Place this file at: ml_training/train_models.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
VISUALIZATIONS_DIR = os.path.join(ML_MODELS_DIR, 'visualizations')

# Create directories
os.makedirs(ML_MODELS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {title}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {filename}")


def plot_learning_curve(estimator, X, y, title, filename, cv=5):
    """Plot and save learning curve"""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy' if hasattr(estimator, 'predict_proba') else 'r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation Score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='g')
    
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Learning Curve - {title}', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Learning curve saved: {filename}")


def plot_feature_importance(model, feature_names, title, filename, top_n=15):
    """Plot and save feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Feature Importance - {title}', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature importance saved: {filename}")


def plot_regression_results(y_true, y_pred, title, filename):
    """Plot actual vs predicted for regression"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=50)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Actual vs Predicted - {title}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Regression plot saved: {filename}")


def train_bmi_classifier():
    """Train BMI Classification Model with Enhanced Analytics"""
    print("\n" + "="*70)
    print("TRAINING BMI CLASSIFICATION MODEL")
    print("="*70)
    
    # Load dataset
    bmi_df = pd.read_csv(os.path.join(DATASETS_DIR, 'bmi.csv'))
    obesity_df = pd.read_csv(os.path.join(DATASETS_DIR, 'ObesityDataSet.csv'))
    
    print(f"✓ Loaded {len(bmi_df)} BMI records")
    print(f"✓ Loaded {len(obesity_df)} obesity records")
    
    # Prepare BMI data
    bmi_df['BMI'] = bmi_df['Weight'] / ((bmi_df['Height'] / 100) ** 2)
    
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 25:
            return 'Normal'
        elif 25 <= bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    bmi_df['Category'] = bmi_df['BMI'].apply(categorize_bmi)
    
    # Prepare features
    le_gender = LabelEncoder()
    bmi_df['Gender_Encoded'] = le_gender.fit_transform(bmi_df['Gender'])
    
    X = bmi_df[['Gender_Encoded', 'Height', 'Weight']].values
    y = bmi_df['Category'].values
    
    le_category = LabelEncoder()
    y_encoded = le_category.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Train model with optimized parameters
    print("\n⚙ Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Model Accuracy: {accuracy*100:.2f}%")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
    print(f"✓ Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    
    # Classification Report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_pred, 
                                target_names=le_category.classes_,
                                digits=4))
    
    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, le_category.classes_,
                         'BMI Classification', 'bmi_confusion_matrix.png')
    
    # Plot Learning Curve
    plot_learning_curve(model, X, y_encoded, 
                       'BMI Classification', 'bmi_learning_curve.png')
    
    # Plot Feature Importance
    feature_names = ['Gender', 'Height (cm)', 'Weight (kg)']
    plot_feature_importance(model, feature_names, 
                           'BMI Classification', 'bmi_feature_importance.png')
    
    # Save model and encoders
    with open(os.path.join(ML_MODELS_DIR, 'bmi_classifier.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(ML_MODELS_DIR, 'label_encoder_category.pkl'), 'wb') as f:
        pickle.dump(le_category, f)
    with open(os.path.join(ML_MODELS_DIR, 'label_encoder_gender.pkl'), 'wb') as f:
        pickle.dump(le_gender, f)
    
    print("\n✓ Model and encoders saved successfully!")
    
    return model, accuracy


def train_tdee_regressor():
    """Train TDEE Regression Model with Enhanced Analytics"""
    print("\n" + "="*70)
    print("TRAINING TDEE REGRESSION MODEL")
    print("="*70)
    
    # Create synthetic TDEE dataset
    np.random.seed(42)
    n_samples = 5000
    
    age = np.random.randint(18, 70, n_samples)
    gender = np.random.choice([0, 1], n_samples)  # 0: Female, 1: Male
    weight = np.random.uniform(45, 120, n_samples)
    height = np.random.uniform(150, 200, n_samples)
    activity = np.random.choice([1.2, 1.375, 1.55, 1.725, 1.9], n_samples)
    
    # Calculate BMR using Mifflin-St Jeor Equation
    bmr = np.where(
        gender == 1,
        10 * weight + 6.25 * height - 5 * age + 5,
        10 * weight + 6.25 * height - 5 * age - 161
    )
    
    # Calculate TDEE
    tdee = bmr * activity + np.random.normal(0, 50, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Weight': weight,
        'Height': height,
        'Activity': activity,
        'TDEE': tdee
    })
    
    print(f"✓ Generated {len(df)} synthetic TDEE samples")
    
    # Prepare features
    X = df[['Age', 'Gender', 'Weight', 'Height', 'Activity']].values
    y = df['TDEE'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Train model
    print("\n⚙ Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n✓ R² Score: {r2:.4f}")
    print(f"✓ RMSE: {rmse:.2f} calories")
    print(f"✓ MAE: {mae:.2f} calories")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"✓ Cross-Validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Plot Regression Results
    plot_regression_results(y_test, y_pred, 
                           'TDEE Prediction', 'tdee_regression_plot.png')
    
    # Plot Learning Curve
    plot_learning_curve(model, X, y, 
                       'TDEE Regression', 'tdee_learning_curve.png')
    
    # Plot Feature Importance
    feature_names = ['Age', 'Gender', 'Weight (kg)', 'Height (cm)', 'Activity Level']
    plot_feature_importance(model, feature_names, 
                           'TDEE Regression', 'tdee_feature_importance.png')
    
    # Save model
    with open(os.path.join(ML_MODELS_DIR, 'tdee_regressor.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print("\n✓ Model saved successfully!")
    
    return model, r2


def generate_model_summary():
    """Generate a summary report of all models"""
    print("\n" + "="*70)
    print("GENERATING MODEL SUMMARY REPORT")
    print("="*70)
    
    summary = {
        'BMI Classifier': {
            'Type': 'Random Forest Classification',
            'Features': ['Gender', 'Height', 'Weight'],
            'Classes': ['Underweight', 'Normal', 'Overweight', 'Obese'],
            'Visualizations': [
                'bmi_confusion_matrix.png',
                'bmi_learning_curve.png',
                'bmi_feature_importance.png'
            ]
        },
        'TDEE Regressor': {
            'Type': 'Random Forest Regression',
            'Features': ['Age', 'Gender', 'Weight', 'Height', 'Activity Level'],
            'Target': 'Daily Calorie Expenditure (TDEE)',
            'Visualizations': [
                'tdee_regression_plot.png',
                'tdee_learning_curve.png',
                'tdee_feature_importance.png'
            ]
        }
    }
    
    # Save summary as text file
    summary_path = os.path.join(ML_MODELS_DIR, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ML MODEL TRAINING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        for model_name, info in summary.items():
            f.write(f"\n{model_name}\n")
            f.write("-" * 70 + "\n")
            for key, value in info.items():
                if isinstance(value, list):
                    f.write(f"{key}:\n")
                    for item in value:
                        f.write(f"  - {item}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
    
    print(f"✓ Summary report saved: model_summary.txt")


def main():
    """Main training pipeline"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "ENHANCED ML TRAINING PIPELINE" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # Train BMI Classifier
        bmi_model, bmi_accuracy = train_bmi_classifier()
        
        # Train TDEE Regressor
        tdee_model, tdee_r2 = train_tdee_regressor()
        
        # Generate Summary
        generate_model_summary()
        
        # Final Summary
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n✓ BMI Classifier Accuracy: {bmi_accuracy*100:.2f}%")
        print(f"✓ TDEE Regressor R²: {tdee_r2:.4f}")
        print(f"\n✓ Models saved in: {ML_MODELS_DIR}")
        print(f"✓ Visualizations saved in: {VISUALIZATIONS_DIR}")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()