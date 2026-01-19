from django.urls import path
from . import views

urlpatterns = [
    # Public pages
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    
    # Application pages
    path('apply/', views.apply_for_credit, name='apply_for_credit'),
    path('application/<uuid:application_id>/', views.application_result, name='application_result'),
    path('history/', views.my_applications, name='my_applications'),
    
    # Admin pages
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('train-model/', views.train_model, name='train_model'),
    path('upload-dataset/', views.upload_dataset, name='upload_dataset'),
    path('review-application/<uuid:application_id>/', views.review_application, name='review_application'),
    
    # API endpoints
    path('api/predict/', views.api_predict_credit, name='api_predict'),
      path('test-static/', views.test_static, name='test_static'),
]