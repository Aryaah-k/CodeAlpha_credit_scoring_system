from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import CreditApplication

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']

class CreditApplicationForm(forms.ModelForm):
    class Meta:
        model = CreditApplication
        exclude = [
            'user', 'prediction_score', 'predicted_credit_rating',
            'predicted_status', 'final_decision', 'final_credit_rating',
            'decision_reason', 'submitted_at', 'reviewed_at', 'decision_date',
            'debt_to_income_ratio', 'credit_utilization_ratio', 'employment_duration'
        ]
        widgets = {
            'loan_purpose': forms.Textarea(attrs={'rows': 3}),
            'submitted_at': forms.HiddenInput(),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add CSS classes to form fields
        for field in self.fields:
            self.fields[field].widget.attrs.update({'class': 'form-control'})

    def clean_annual_income(self):
        income = self.cleaned_data.get('annual_income')
        if income is not None and income <= 0:
            raise forms.ValidationError("Annual income must be greater than 0.")
        return income

    def clean_credit_card_limit(self):
        limit = self.cleaned_data.get('credit_card_limit')
        if limit is not None and limit < 0:
            raise forms.ValidationError("Credit card limit cannot be negative.")
        return limit

    def clean_credit_card_balance(self):
        balance = self.cleaned_data.get('credit_card_balance')
        if balance is not None and balance < 0:
            raise forms.ValidationError("Credit card balance cannot be negative.")
        return balance

    def clean_loan_amount(self):
        amount = self.cleaned_data.get('loan_amount')
        if amount is not None and amount <= 0:
            raise forms.ValidationError("Loan amount must be greater than 0.")
        return amount

class ModelTrainingForm(forms.Form):
    model_choice = forms.ChoiceField(
        choices=[
            ('random_forest', 'Random Forest'),
            ('logistic_regression', 'Logistic Regression'),
            ('decision_tree', 'Decision Tree'),
            ('gradient_boosting', 'Gradient Boosting'),
            ('all', 'All Models')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    dataset_file = forms.FileField(
        label='Upload Dataset (CSV)',
        widget=forms.FileInput(attrs={'class': 'form-control'}),
        required=False
    )
    
    perform_hyperparameter_tuning = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

class DatasetUploadForm(forms.Form):
    dataset_file = forms.FileField(
        label='Credit Dataset (CSV)',
        widget=forms.FileInput(attrs={'class': 'form-control'}),
        help_text='Upload a CSV file with credit scoring data'
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        required=False
    )