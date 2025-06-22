from django.urls import path
from .views import (
  EmotionAnalysisView,
)

urlpatterns = [   
   # Emotion analysis
    path('analyze/', EmotionAnalysisView.as_view(), name='emotion-analyze'),
]