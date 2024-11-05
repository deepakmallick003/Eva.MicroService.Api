import json
import requests
from scripts.models import model_analytics
from core.config import settings

class Analytics:
    def __init__(self):
        pass
    
    def store_analytics(self, payload: model_analytics.AnalyticsRequest):
        url = settings.CABI_AnalyticsCollector_API_BaseURL
        headers = {'Content-Type': 'application/json'}
        
        if hasattr(payload, "model_dump"):
            payload_dict = payload.model_dump() 
        else:
            payload_dict = json.loads(json.dumps(payload, default=lambda o: o.__dict__))

        payload_json = json.dumps(payload_dict) 
        response = requests.post(url, headers=headers, json=payload_json)
        
        try:
            response.raise_for_status() 
            response_text = response.text
            return response_text
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
