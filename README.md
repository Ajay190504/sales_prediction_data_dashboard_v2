# Sales Prediction & Data Visualization

## Overview

Sales Prediction & Data Visualization is a machine learning and analytics platform designed to transform historical sales data into actionable business insights. The application combines data preprocessing, exploratory data analysis (EDA), interactive visualizations, and predictive modeling to help businesses understand past performance and forecast future sales trends.

The system enables users to upload datasets, analyze sales patterns, visualize key metrics, and generate future sales predictions using machine learning algorithms.

---

## Features

### Data Processing
- Upload CSV and Excel datasets
- Automated data cleaning and preprocessing
- Missing value handling
- Duplicate record removal
- Data validation and formatting

### Data Analysis
- Exploratory Data Analysis (EDA)
- Statistical summaries
- Trend analysis
- Correlation analysis

### Interactive Visualizations
- Monthly sales trends
- Revenue analysis
- Product-wise performance charts
- Region-wise sales distribution
- Customer behavior analysis
- Forecast visualization

### Machine Learning
- Regression-based sales forecasting
- Future sales prediction
- Model performance evaluation

### Performance Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score

---

## Technology Stack

### Backend
- Python
- Flask

### Data Processing
- Pandas
- NumPy

### Data Visualization
- Plotly
- Matplotlib

### Machine Learning
- Scikit-learn

### Deployment
- Render
- Docker

---

## Project Workflow

1. Load historical sales data
2. Clean and preprocess the dataset
3. Perform exploratory data analysis
4. Generate visual insights
5. Train machine learning models
6. Predict future sales trends
7. Evaluate model performance
8. Display results through dashboards

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Ajay190504/sales_prediction_data_dashboard_v2.git
cd sales_prediction_data_dashboard_v2
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

---

## Deployment on Render

### Build Command

```bash
pip install -r requirements-render.txt
```

### Start Command

```bash
gunicorn -w 4 -b 0.0.0.0:$PORT app:app
```

### Environment Variable

```text
FLASK_ENV=production
```

---

## Docker Deployment

### Build Image

```bash
docker build -t sales-pred-app .
```

### Run Container

```bash
docker run -p 5000:5000 --rm -e FLASK_ENV=production sales-pred-app
```

---

## Applications

- Sales Forecasting
- Revenue Analysis
- Inventory Planning
- Demand Prediction
- Business Intelligence
- Financial Reporting
- Marketing Strategy Optimization

---

## Future Enhancements

- Deep Learning Forecasting Models
- Real-Time Data Streaming
- Advanced Business Intelligence Reports
- Multi-User Support
- Cloud Database Integration
- AI-Powered Sales Recommendations

---

## Screenshots

### Dashboard
Add dashboard screenshot here

### Sales Trends
Add sales trend chart screenshot here

### Forecasting Results
Add prediction results screenshot here

### Data Analysis
Add EDA screenshots here

---

## Author

**Ajay D. Waghmare**

B.Tech Computer Science & Engineering

Java Full Stack Developer | Machine Learning Enthusiast | Data Analytics Practitioner
