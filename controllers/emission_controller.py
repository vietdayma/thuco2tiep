from models.emission_model import EmissionModel
import pandas as pd
import requests
import os

class EmissionController:
    def __init__(self):
        self.model = EmissionModel()
        self.trained = False
        self.avg_emission = None
        # Use API URL from environment or default to local for development
        self.api_url = os.environ.get('API_URL', 'http://localhost:10000') + "/predict"

    def initialize_model(self, data_path):
        """Initialize and train the model"""
        test_score = self.model.train(data_path)
        self.trained = True
        
        # Calculate average emission
        df = self.model.load_and_preprocess_data(data_path)
        self.avg_emission = df['CO2 Emissions(g/km)'].mean()
        
        return test_score

    def predict_emission(self, features):
        """Make prediction using the model"""
        if not self.trained:
            raise ValueError("Model needs to be trained first!")
        
        return self.model.predict(features)

    def predict_emission_api(self, features):
        """Make prediction using the API and return full response including processing time"""
        try:
            # Send request to API
            response = requests.post(self.api_url, json=features)
            response.raise_for_status()
            
            # Return full response data
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.trained:
            raise ValueError("Model needs to be trained first!")
        
        return self.model.get_feature_importance()

    def get_average_emission(self):
        """Get average emission value"""
        return self.avg_emission

    def get_emission_rating(self, emission_value):
        """Get emission rating (A to F)"""
        if emission_value < 100:
            return 'A'
        elif emission_value < 120:
            return 'B'
        elif emission_value < 140:
            return 'C'
        elif emission_value < 160:
            return 'D'
        elif emission_value < 180:
            return 'E'
        else:
            return 'F'

    def get_eco_tips(self, emission_value):
        """Get eco-friendly tips based on emission value"""
        tips = []
        if emission_value > 160:
            tips.extend([
                "Consider switching to a more fuel-efficient vehicle",
                "Regular maintenance can help reduce emissions",
                "Avoid aggressive acceleration and braking"
            ])
        if emission_value > 140:
            tips.extend([
                "Check tire pressure regularly",
                "Remove excess weight from the vehicle"
            ])
        tips.extend([
            "Use eco-driving techniques",
            "Plan your trips to avoid traffic"
        ])
        return tips 