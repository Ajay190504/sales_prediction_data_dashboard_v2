# Sales Prediction & Data Visualization - v2

## 🔍 Project Overview

*Sales Prediction and Data Visualization* is a data analytics and machine learning project that analyzes historical sales data to identify patterns, generate interactive visual insights, and predict future sales trends.  
The system helps businesses make *data-driven decisions* by forecasting revenue, understanding customer behavior, and optimizing sales strategies.

This project combines *data preprocessing, exploratory data analysis (EDA), visualization dashboards, and predictive modeling* to deliver a complete sales intelligence solution.

---

## 🎯 Objectives

- Clean and preprocess raw sales datasets.
- Visualize sales performance using charts and dashboards.
- Analyze trends based on:
  - Time (daily, monthly, yearly)
  - Product categories
  - Regions and customers
- Build machine learning models to *predict future sales*.
- Provide actionable insights to improve business planning.

---

## 🛠️ Technologies Used

| Category            | Tools / Libraries |
|---------------------|------------------|
| Programming Language | Python |
| Data Handling       | Pandas, NumPy |
| Visualization       | Matplotlib, Seaborn, Plotly |
| Machine Learning    | Scikit-learn |
| Environment         | Jupyter Notebook / VS Code |
| Dataset Format      | CSV / Excel |

---

## 📈 Key Features

- 📂 Import and process large sales datasets  
- 🧹 Handle missing values and data inconsistencies  
- 📊 Interactive data visualization dashboards  
- 🔎 Exploratory Data Analysis (EDA)  
- 🤖 Predictive sales model using regression algorithms  
- 📅 Future sales forecasting  
- 📄 Performance evaluation using MAE, RMSE, R²  

---

## ⚙️ Workflow

1. *Data Collection* – Load historical sales dataset  
2. *Data Cleaning* – Remove null values, format columns  
3. *EDA & Visualization* – Generate graphs for insights  
4. *Feature Engineering* – Extract useful features  
5. *Model Training* – Train ML regression models  
6. *Prediction* – Forecast upcoming sales  
7. *Evaluation* – Validate model accuracy  

---

## 📊 Sample Visualizations

- Monthly Sales Trend Line Chart  
- Product-wise Revenue Bar Chart  
- Region-wise Sales Heatmap  
- Sales Forecast Graph  

---

## 🚀 Applications

- Business sales forecasting  
- Inventory planning  
- Marketing strategy optimization  
- Financial performance tracking  

---

## 📌 Conclusion

This project demonstrates the effective use of *data analytics and machine learning* in predicting business sales and visualizing meaningful insights. It provides a powerful tool for organizations to understand past performance and plan for future growth with confidence.

Run:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
Open http://localhost:5000

Deploying to Render (recommended quick path)
-------------------------------------------

1. Commit and push this repository to GitHub (or your preferred Git host).
2. Sign in to https://render.com and create a new Web Service.
3. Connect your GitHub account and pick the repository containing this project.
4. Configure the service:
	- Build Command: pip install -r requirements-render.txt
	- Start Command: gunicorn -w 4 -b 0.0.0.0:$PORT app:app
	- Environment: set `FLASK_ENV=production` (Render sets sensible defaults but set this for clarity)
5. Choose an instance plan (free plans can work for light usage).
6. Create the service and watch the build logs.

Notes:
- The file `requirements-render.txt` is a trimmed dependency list that omits heavy optional packages (for example `xgboost` and `pmdarima`) to reduce build time and memory usage. If you need those features, use `requirements.txt` instead but expect longer builds and larger runtime memory.
- Render's filesystem is ephemeral. Session files written to `sessions/` will not persist across automatic redeploys or machine restarts. For durable storage, configure an external storage (S3, database, or persistent disk) and update `app.py` to use that storage for session artifacts.

A Dockerfile has been added to the repository for reproducible builds. Below are quick Docker build and run instructions you can use locally or in CI. These commands are tailored for PowerShell on Windows.

Build (uses the trimmed `requirements-render.txt` by default):

```powershell
docker build -t sales-pred-app .
```

If you want to build with the full `requirements.txt` instead, run:

```powershell
docker build --build-arg REQ=requirements.txt -t sales-pred-app:full .
```

Run:

```powershell
docker run -p 5000:5000 --rm -e FLASK_ENV=production sales-pred-app
```

Notes on Docker:
- The included `Dockerfile` is based on `python:3.11-slim` and installs dependencies from the chosen requirements file. It exposes port 5000 to match the app's Gunicorn start command.
- The `.dockerignore` excludes `sessions/` and other large or sensitive files to keep the image small. If you require session persistence, mount a host directory or use cloud storage (S3) and update `app.py` accordingly.

If you'd like, I can also add a multi-stage Dockerfile that builds wheels for optional heavy dependencies (xgboost/pmdarima) to speed up deploys, or create a GitHub Actions workflow to build and push container images automatically.
