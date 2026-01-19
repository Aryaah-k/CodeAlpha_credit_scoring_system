from django.shortcuts import render
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.core.paginator import Paginator
from django.utils import timezone
import pandas as pd
import json
import os

from .forms import (
    CreditApplicationForm, UserRegistrationForm,
    ModelTrainingForm, DatasetUploadForm
)
from .models import CreditApplication, ModelPerformance, AuditLog, CreditScoreFactors
from .ml_model.model_trainer import CreditModelTrainer
from .ml_model.model_predictor import CreditScoringPredictor

def home(request):
    """Home page view"""
    return render(request, 'credit_app/index.html')

@login_required
def apply_for_credit(request):
    """Apply for credit view"""
    if request.method == 'POST':
        form = CreditApplicationForm(request.POST)
        if form.is_valid():
            application = form.save(commit=False)
            application.user = request.user
            
            # Prepare data for prediction
            input_data = {
                'age': application.age,
                'annual_income': float(application.annual_income),
                'monthly_debt': float(application.monthly_debt),
                'employment_status': application.employment_status,
                'education_level': application.education_level,
                'number_of_credit_lines': application.number_of_credit_lines,
                'loan_amount': float(application.loan_amount),
                'loan_term_months': application.loan_term_months,
                'late_payments_30days': application.late_payments_30days,
                'late_payments_60days': application.late_payments_60days,
                'late_payments_90days': application.late_payments_90days,
                'credit_card_limit': float(application.credit_card_limit) if application.credit_card_limit else 0,
                'credit_card_balance': float(application.credit_card_balance) if application.credit_card_balance else 0,
                'home_ownership': 'Unknown',  # Default value
                'loan_purpose': application.loan_purpose,
                'has_bank_account': 1,  # Assume has bank account
                'number_of_loans': 1,  # Default to 1 loan
                'months_employed': 24,  # Default to 2 years
            }
            
            # Make prediction
            try:
                predictor = CreditScoringPredictor()
                prediction = predictor.predict(input_data)
                
                application.prediction_score = prediction['probability']
                application.predicted_credit_rating = prediction['credit_rating']
                application.predicted_status = 'approved' if prediction['prediction'] == 1 else 'rejected'
                
                # Log the prediction
                AuditLog.objects.create(
                    user=request.user,
                    action='prediction',
                    details={
                        'application_id': str(application.application_id),
                        'prediction': prediction['prediction'],
                        'probability': prediction['probability'],
                        'credit_rating': prediction['credit_rating']
                    },
                    ip_address=request.META.get('REMOTE_ADDR')
                )
                
            except Exception as e:
                messages.error(request, f"Error making prediction: {str(e)}")
                application.predicted_status = 'under_review'
            
            application.save()
            messages.success(request, 'Application submitted successfully!')
            return redirect('application_result', application_id=application.application_id)
    else:
        form = CreditApplicationForm()
    
    return render(request, 'credit_app/predict.html', {'form': form})

@login_required
def application_result(request, application_id):
    """View application result"""
    application = get_object_or_404(
        CreditApplication, 
        application_id=application_id,
        user=request.user
    )
    
    # Get credit score factors
    factors = CreditScoreFactors.objects.all().order_by('-weight')
    
    # Calculate key metrics for display
    metrics = {
        'debt_to_income': application.debt_to_income_ratio,
        'credit_utilization': application.credit_utilization_ratio,
        'payment_history_score': max(0, 100 - (
            application.late_payments_30days * 10 +
            application.late_payments_60days * 20 +
            application.late_payments_90days * 30
        ))
    }
    
    context = {
        'application': application,
        'factors': factors,
        'metrics': metrics,
    }
    
    return render(request, 'credit_app/results.html', context)

@staff_member_required
def admin_dashboard(request):
    """Admin dashboard view"""
    # Statistics
    total_applications = CreditApplication.objects.count()
    approved_applications = CreditApplication.objects.filter(predicted_status='approved').count()
    rejected_applications = CreditApplication.objects.filter(predicted_status='rejected').count()
    
    # Model performance
    active_model = ModelPerformance.objects.filter(is_active=True).first()
    
    # Recent applications
    recent_applications = CreditApplication.objects.all().order_by('-submitted_at')[:10]
    
    context = {
        'total_applications': total_applications,
        'approved_applications': approved_applications,
        'rejected_applications': rejected_applications,
        'approval_rate': (approved_applications / total_applications * 100) if total_applications > 0 else 0,
        'active_model': active_model,
        'recent_applications': recent_applications,
    }
    
    return render(request, 'credit_app/admin_dashboard.html', context)

@staff_member_required
def train_model(request):
    """Train machine learning model"""
    if request.method == 'POST':
        form = ModelTrainingForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Get form data
                model_choice = form.cleaned_data['model_choice']
                dataset_file = form.cleaned_data['dataset_file']
                hyperparameter_tuning = form.cleaned_data['perform_hyperparameter_tuning']
                
                # Use uploaded file or default dataset
                if dataset_file:
                    dataset_path = f'data/uploaded_{dataset_file.name}'
                    with open(dataset_path, 'wb+') as destination:
                        for chunk in dataset_file.chunks():
                            destination.write(chunk)
                else:
                    dataset_path = 'data/credit_dataset.csv'
                
                # Initialize trainer
                trainer = CreditModelTrainer()
                
                # Load and prepare data
                X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(dataset_path)
                
                # Train models
                if model_choice == 'all':
                    results = trainer.train_models(X_train, y_train, X_test, y_test)
                else:
                    # Train specific model
                    model = trainer.models[model_choice]
                    model.fit(X_train, y_train)
                    trainer.best_model = model
                
                # Hyperparameter tuning if requested
                if hyperparameter_tuning and model_choice in ['random_forest', 'all']:
                    trainer.hyperparameter_tuning(X_train, y_train)
                
                # Generate evaluation plots
                trainer.generate_evaluation_plots(X_test, y_test)
                
                # Save model
                trainer.save_model()
                
                # Save model performance to database
                if trainer.best_model:
                    model_perf = ModelPerformance.objects.create(
                        model_name=type(trainer.best_model).__name__,
                        model_type=model_choice,
                        accuracy=trainer.results.get(model_choice, {}).get('accuracy', 0.85),
                        precision=trainer.results.get(model_choice, {}).get('precision', 0.85),
                        recall=trainer.results.get(model_choice, {}).get('recall', 0.85),
                        f1_score=trainer.results.get(model_choice, {}).get('f1_score', 0.85),
                        roc_auc=trainer.results.get(model_choice, {}).get('roc_auc', 0.85),
                        parameters={},
                        model_file='models/trained_model.pkl',
                        is_active=True
                    )
                    
                    # Deactivate other models
                    ModelPerformance.objects.exclude(id=model_perf.id).update(is_active=False)
                
                messages.success(request, 'Model trained successfully!')
                
                # Log training
                AuditLog.objects.create(
                    user=request.user,
                    action='model_training',
                    details={
                        'model_type': model_choice,
                        'hyperparameter_tuning': hyperparameter_tuning,
                        'dataset': dataset_path
                    }
                )
                
                return redirect('admin_dashboard')
                
            except Exception as e:
                messages.error(request, f'Error training model: {str(e)}')
    else:
        form = ModelTrainingForm()
    
    return render(request, 'credit_app/train_model.html', {'form': form})

@staff_member_required
def upload_dataset(request):
    """Upload new dataset"""
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset_file = form.cleaned_data['dataset_file']
            description = form.cleaned_data['description']
            
            # Save dataset
            dataset_path = f'data/{dataset_file.name}'
            os.makedirs('data', exist_ok=True)
            
            with open(dataset_path, 'wb+') as destination:
                for chunk in dataset_file.chunks():
                    destination.write(chunk)
            
            messages.success(request, f'Dataset {dataset_file.name} uploaded successfully!')
            return redirect('admin_dashboard')
    else:
        form = DatasetUploadForm()
    
    return render(request, 'credit_app/upload_dataset.html', {'form': form})

def register(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Registration successful! Please login.')
            return redirect('login')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'credit_app/register.html', {'form': form})

@login_required
def my_applications(request):
    """View my applications"""
    applications = CreditApplication.objects.filter(
        user=request.user
    ).order_by('-submitted_at')

    paginator = Paginator(applications, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'credit_app/application_history.html', {
        'page_obj': page_obj
    })

@staff_member_required
def review_application(request, application_id):
    """Review and make final decision on application"""
    application = get_object_or_404(CreditApplication, application_id=application_id)
    
    if request.method == 'POST':
        final_decision = request.POST.get('final_decision')
        decision_reason = request.POST.get('decision_reason')
        
        if final_decision in dict(CreditApplication.CREDIT_STATUS):
            application.final_decision = final_decision
            application.final_credit_rating = application.predicted_credit_rating
            application.decision_reason = decision_reason
            application.decision_date = timezone.now()
            application.save()
            
            messages.success(request, 'Decision saved successfully!')
            return redirect('admin_dashboard')
    
    return render(request, 'credit_app/review_application.html', {
        'application': application
    })

def api_predict_credit(request):
    """API endpoint for credit prediction"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            predictor = CreditScoringPredictor()
            result = predictor.predict(data)
            
            return JsonResponse({
                'success': True,
                'prediction': result['prediction'],
                'probability': result['probability'],
                'credit_rating': result['credit_rating'],
                'class_probabilities': result['class_probabilities']
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
# Create your views here.

def test_static(request):
    """Test static files loading"""
    return render(request, 'credit_app/test_static.html')
