from django.db import models
from django.contrib.auth.models import User
import uuid

class CreditApplication(models.Model):
    CREDIT_STATUS = [
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
        ('pending', 'Pending'),
        ('under_review', 'Under Review')
    ]
    
    CREDIT_RATINGS = [
        ('A+', 'Excellent (A+)'),
        ('A', 'Very Good (A)'),
        ('B', 'Good (B)'),
        ('C', 'Fair (C)'),
        ('D', 'Poor (D)'),
        ('E', 'Very Poor (E)')
    ]
    
    application_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='credit_applications')
    
    # Personal Information
    age = models.IntegerField()
    annual_income = models.DecimalField(max_digits=12, decimal_places=2)
    monthly_debt = models.DecimalField(max_digits=10, decimal_places=2)
    employment_status = models.CharField(max_length=50)
    employment_duration = models.IntegerField(default=0, help_text="In months")
    education_level = models.CharField(max_length=50)
    
    # Credit Information
    credit_score = models.IntegerField(null=True, blank=True)
    number_of_credit_lines = models.IntegerField()
    credit_card_limit = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    credit_card_balance = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    credit_utilization_ratio = models.FloatField(null=True, blank=True)
    
    # Loan Information
    loan_amount = models.DecimalField(max_digits=10, decimal_places=2)
    loan_purpose = models.CharField(max_length=200)
    loan_term_months = models.IntegerField()
    interest_rate = models.FloatField(null=True, blank=True)
    
    # Payment History
    late_payments_90days = models.IntegerField(default=0)
    late_payments_60days = models.IntegerField(default=0)
    late_payments_30days = models.IntegerField(default=0)
    
    # Model Prediction
    prediction_score = models.FloatField(null=True, blank=True)
    predicted_credit_rating = models.CharField(max_length=10, choices=CREDIT_RATINGS, null=True, blank=True)
    predicted_status = models.CharField(max_length=20, choices=CREDIT_STATUS, default='pending')
    
    # Calculated Fields
    debt_to_income_ratio = models.FloatField(null=True, blank=True)
    
    # Timestamps
    submitted_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    decision_date = models.DateTimeField(null=True, blank=True)
    
    # Final Decision
    final_decision = models.CharField(max_length=20, choices=CREDIT_STATUS, null=True, blank=True)
    final_credit_rating = models.CharField(max_length=10, choices=CREDIT_RATINGS, null=True, blank=True)
    decision_reason = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-submitted_at']
    
    def save(self, *args, **kwargs):
        # Calculate debt-to-income ratio
        if self.annual_income and self.monthly_debt and float(self.annual_income) > 0:
            self.debt_to_income_ratio = (float(self.monthly_debt) * 12) / float(self.annual_income)

        # Calculate credit utilization ratio
        if self.credit_card_limit and self.credit_card_balance and float(self.credit_card_limit) > 0:
            self.credit_utilization_ratio = float(self.credit_card_balance) / float(self.credit_card_limit)

        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Application {self.application_id} - {self.user.username}"

class ModelPerformance(models.Model):
    model_name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=50)
    training_date = models.DateTimeField(auto_now_add=True)
    
    # Performance Metrics
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    roc_auc = models.FloatField()
    
    # Model Configuration
    parameters = models.JSONField()
    feature_importance = models.JSONField(null=True, blank=True)
    
    # Model File
    model_file = models.FileField(upload_to='models/')
    
    # Active Status
    is_active = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-training_date']
    
    def __str__(self):
        return f"{self.model_name} - Accuracy: {self.accuracy:.4f}"

class CreditScoreFactors(models.Model):
    factor_name = models.CharField(max_length=100)
    description = models.TextField()
    weight = models.FloatField(help_text="Importance weight in scoring")
    impact = models.CharField(max_length=20, choices=[
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral', 'Neutral')
    ])
    
    def __str__(self):
        return f"{self.factor_name} (Weight: {self.weight})"

class AuditLog(models.Model):
    ACTION_CHOICES = [
        ('prediction', 'Prediction Made'),
        ('model_training', 'Model Training'),
        ('application_submission', 'Application Submission'),
        ('decision_made', 'Decision Made'),
        ('system_change', 'System Change')
    ]
    
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    details = models.JSONField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.timestamp} - {self.action} by {self.user}"

# Create your models here.
