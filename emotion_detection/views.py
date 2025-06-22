from django.http import JsonResponse
from django.views import View
import json
import os
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .ml_model import EmotionDetector

# Load the model once when the server starts
# Make sure the model file is in the right path
MODEL_PATH = r"D:\Esewa\Project for esewa\my_emotion_analysis\model_implementation\emotion_detection.pkl"
detector = EmotionDetector(model_path=MODEL_PATH)

@method_decorator(csrf_exempt, name='dispatch')
class EmotionAnalysisView(View):
    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            text = data.get("text")

            if not text:
                return JsonResponse({"error": "No text provided."}, status=400)

            result = detector.predict(text)

            return JsonResponse({"emotion": result[0]})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
