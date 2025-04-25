import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_feature_importance(importance_dict):
    """Plot feature importance scores"""
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': importance_dict.keys(),
        'Importance': importance_dict.values()
    }).sort_values('Importance', ascending=True)
    
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
    plt.title('Feature Importance in CO2 Emission Prediction')
    return fig

def plot_emission_comparison(prediction, avg_emission):
    """Plot prediction vs average emission"""
    fig, ax = plt.subplots(figsize=(8, 6))
    emissions = [avg_emission, prediction]
    labels = ['Average Emission', 'Predicted Emission']
    colors = ['lightgray', 'lightgreen' if prediction < avg_emission else 'lightcoral']
    
    ax.bar(labels, emissions, color=colors)
    plt.title('CO2 Emission Comparison')
    plt.ylabel('CO2 Emissions (g/km)')
    
    # Add value labels on top of bars
    for i, v in enumerate(emissions):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    return fig

def create_gauge_chart(value, min_val, max_val, title):
    """Create a gauge chart for emissions"""
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'polar'})
    
    # Convert value to angle
    angle = (value - min_val) / (max_val - min_val) * np.pi
    
    # Create the gauge
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    
    # Plot the gauge
    ax.plot([0, angle], [0, 0.9], color='red', linewidth=3)
    
    # Customize the chart
    ax.set_rticks([])
    ax.set_xticks(np.linspace(0, np.pi, 5))
    ax.set_xticklabels([f'{v:.0f}' for v in np.linspace(min_val, max_val, 5)])
    
    plt.title(title)
    return fig

def style_metric_cards():
    """Return CSS styling for metric cards"""
    return """
    <style>
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            color: #0f4c81;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #1f77b4;
        }
    </style>
    """ 