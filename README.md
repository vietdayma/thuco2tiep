# CO2 Emission Predictor

An advanced web application for predicting and analyzing vehicle CO2 emissions using machine learning.

## Features

- Predict CO2 emissions based on vehicle specifications
- Analyze feature importance in emission predictions
- Get eco-friendly tips based on emission levels
- Visual comparisons with average emissions
- Emission rating system (A to F)
- Interactive gauge charts and visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/co2-emission-predictor.git
cd co2-emission-predictor
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have the dataset file `co2 Emissions.csv` in the project root directory.

2. Run the application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501).

## Project Structure

```
co2-emission-predictor/
├── app.py                  # Main application file
├── models/                 # Model-related code
│   └── emission_model.py
├── views/                  # View-related code
│   └── main_view.py
├── controllers/            # Controller-related code
│   └── emission_controller.py
├── utils/                  # Utility functions
│   └── visualization.py
├── static/                 # Static files
│   └── images/
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Model Features

The model takes into account the following vehicle specifications:
- Engine Size (L)
- Number of Cylinders
- Fuel Consumption (L/100 km)
- Horsepower
- Vehicle Weight (kg)
- Vehicle Year

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 