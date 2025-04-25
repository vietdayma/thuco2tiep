import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class EmissionModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.features = [
            'Engine Size(L)', 
            'Cylinders', 
            'Fuel Consumption Comb (L/100 km)',
            'Horsepower',  # New feature
            'Weight (kg)', # New feature
            'Year'        # New feature
        ]
        self.target = 'CO2 Emissions(g/km)'
        self.trained = False

    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the dataset"""
        df = pd.read_csv(data_path)
        
        # Add synthetic features for demonstration
        np.random.seed(42)
        df['Horsepower'] = df['Engine Size(L)'] * 100 + np.random.normal(0, 10, len(df))
        df['Weight (kg)'] = df['Engine Size(L)'] * 500 + np.random.normal(0, 50, len(df))
        df['Year'] = np.random.randint(2015, 2024, len(df))
        
        # Map fuel types
        fuel_type_mapping = {
            "Z": "Premium Gasoline",
            "X": "Regular Gasoline",
            "D": "Diesel",
            "E": "Ethanol(E85)",
            "N": "Natural Gas"
        }
        df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)
        
        # Remove natural gas vehicles (too few samples)
        df = df[~df["Fuel Type"].str.contains("Natural Gas")].reset_index(drop=True)
        
        return df

    def prepare_features(self, df):
        """Prepare features for training/prediction"""
        X = df[self.features].copy()
        if self.target in df.columns:
            y = df[self.target]
        else:
            y = None
        return X, y

    def train(self, data_path):
        """Train the model"""
        df = self.load_and_preprocess_data(data_path)
        X, y = self.prepare_features(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.trained = True
        
        # Calculate and return metrics
        X_test_scaled = self.scaler.transform(X_test)
        test_score = self.model.score(X_test_scaled, y_test)
        return test_score

    def predict(self, features_dict):
        """Make predictions"""
        if not self.trained:
            raise ValueError("Model needs to be trained first!")
            
        # Convert input dictionary to DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Scale the features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction

    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.trained:
            raise ValueError("Model needs to be trained first!")
            
        importance_dict = dict(zip(self.features, self.model.feature_importances_))
        return importance_dict 