# nisr_hackathon_dashboard
Ending Hidden Hunger Dashboard
A comprehensive Streamlit application for analyzing and visualizing malnutrition data in Rwanda to support evidence-based policy decisions and interventions.

Overview
The Ending Hidden Hunger Dashboard is a data-driven tool designed to analyze household survey data and identify patterns of malnutrition across Rwanda. It provides interactive visualizations, predictive modeling, and policy recommendations to help stakeholders address food insecurity and malnutrition effectively.

Features
Dashboard Overview: Key metrics and visualizations of malnutrition indicators
Data Exploration: Comprehensive data overview with statistics and distributions
Geographic Analysis: Interactive maps identifying malnutrition hotspots
Predictive Modeling: Machine learning models to predict malnutrition risk
Root Cause Analysis: Analysis of factors contributing to malnutrition
Policy Recommendations: Evidence-based recommendations for interventions
Installation
Clone this repository:
bash

Line Wrapping

Collapse
Copy
1
2
git clone https://github.com/methodewan/nisr_hackathon_dashboard.git
git clone 
cd ending-hidden-hunger-dashboard
Install the required packages:
bash

Line Wrapping

Collapse
Copy
1
pip install -r requirements.txt
Run the application:
bash

Line Wrapping

Collapse
Copy
1
streamlit run app.py
The application will be available at http://localhost:8501 by default.

Requirements
Python 3.8+
Streamlit
Pandas
NumPy
Plotly
Matplotlib
Seaborn
Scikit-learn
Usage
Dashboard: View key metrics and overall malnutrition indicators
Data Overview: Explore the dataset and understand variable distributions
Malnutrition Hotspots: Identify geographic areas with high malnutrition risk
Predictive Models: Use machine learning to predict malnutrition risk
Root Cause Analysis: Analyze factors contributing to malnutrition
Policy Recommendations: Review evidence-based interventions
Data
The application uses household survey data from the CFSVA2024 survey. The data includes information on:

Household demographics
Food consumption patterns
Coping strategies
Wealth indicators
Geographic information
The application can work with both real CSV data and mock data for demonstration purposes.

File Structure

ending-hidden-hunger-dashboard/
├── app.py                 # Main application file
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── data/                  # Data directory (optional)
    └── CFSVA2024_HH_data.csv  # Sample data file
Configuration
The application can be configured by modifying the following in app.py:

Data source path
Visualization parameters
Model parameters
UI elements
Contributing
Fork the repository
Create a feature branch
Make your changes
Submit a pull request
License
This project is licensed under the MIT License.

Contact
For questions or support, please contact:

Email: your.email@example.com
GitHub: https://github.com/yourusername/ending-hidden-hunger-dashboard
Acknowledgments
Data source: CFSVA2024 Household Survey
Built with Streamlit
Visualizations powered by Plotly
Machine learning with Scikit-learn
