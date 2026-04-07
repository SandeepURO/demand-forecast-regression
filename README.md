# Demand Forecast Regression

Machine Learning project that predicts product demand using regression algorithms including **Linear Regression, Random Forest, and Gradient Boosting**.  
The system analyzes historical order data to forecast future demand and provides predictions through a **Flask API and web interface**.

---

## Project Overview

Demand forecasting is an important problem in supply chain management.  
This project builds a machine learning pipeline to predict product demand based on historical order data.

The project compares multiple regression models and evaluates their performance using standard machine learning metrics.

---

## Models Used

- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

Random Forest achieved the best performance in the final model.

---

## Technologies Used

### Programming Languages

- Python  
- JavaScript  

### Machine Learning

- Scikit-learn  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  

### Backend

- Flask API  

### Frontend

- HTML  
- CSS  
- JavaScript  

---

## Project Structure


demand-forecast-regression
│
├── backend
│ └── app.py
│
├── data
│ └── Historical Product Demand.csv
│
├── model
│ └── scaler.pkl
│
├── notebooks
│
├── deployment
│
├── docs
│
├── static
├── templates
├── netlify_site
│
├── README.md
├── requirements.txt
└── netlify.toml


---

## Machine Learning Pipeline

1. Data cleaning and preprocessing  
2. Feature engineering  
3. Model training  
4. Model comparison  
5. Performance evaluation  
6. Deployment through Flask API  

---

## Evaluation Metrics

- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- R² Score  

---

## Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
Run the Flask API
python backend/app.py
Open in browser
http://127.0.0.1:5000
```


  Future Improvements
Deep learning forecasting models
Real-time demand prediction
Advanced feature engineering
Deployment using Docker and cloud services
Author

Purohit Sandeep Kumar
B.Tech Computer Science Engineering (AI & ML)
Sir Padampat Singhania University

GitHub:
https://github.com/SandeepURO
