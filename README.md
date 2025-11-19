# FINAL PROJECT PROPOSAL (DATS-6103)

**PROJECT NAME:** Machine Learning for Kubernetes Container Load Prediction and Scaling  
**TEAM MEMBERS:** Aditi Shukla, Ashley Gyapomah  
**SUBMISSION DEADLINE:** Oct 24  

---

## SMART QUESTION

**RESEARCH QUESTION:** How can machine learning predict container overload and improve Kubernetes scaling performance?

**S:** Analyze CPU, memory, latency, and configuration features that drive overload events.  
**M:** Evaluate ML models using Accuracy, Precision, Recall, F1-Score, ROC-AUC.  
**A:** Use Kaggle’s Kubernetes Resource & Performance Metrics dataset.  
**R:** Supports proactive scaling and improves cluster reliability.  
**T:** Completed over a 4-week structured timeline.

---

## OBJECTIVE
**OBJECTIVE:** Predict upcoming container overloads in Kubernetes before autoscaling events occur.

---

## PROBLEM STATEMENT
In Kubernetes clusters, containers often exceed CPU/memory limits before Kubernetes creates new pods.  
This scaling delay causes:
- Buffering  
- Slow responses  
- Temporary performance issues  

This project aims to predict overload **before** it happens, enabling proactive scaling.

---

## SOLUTION
Using the Kubernetes resource + performance metrics dataset, we will:
- Detect patterns that indicate overload  
- Train ML models to forecast when scaling is needed  
- Reduce delay between load spikes and autoscaling  
- Improve system responsiveness  

---

## IMPACT
- Triggers early scaling instead of waiting for failures  
- Avoids temporary lag or downtime  
- Enhances Kubernetes autoscaling efficiency  
- Improves user experience under high load  

---

## DATASET

**SOURCE:** Kaggle – Kubernetes Resource & Performance Metrics Allocation Dataset  

**CONTAINS TWO CATEGORIES:**  
- **Performance Metrics:** CPU usage, memory usage, disk I/O, network latency  
- **Resource Allocation Metrics:** CPU/memory limits, requests, scaling configs  

**COVERS DATA FROM:** Pods and nodes across Kubernetes clusters  

**PURPOSE:** Analyze resource behavior and predict overload patterns  

**USE CASE:** Build ML models to detect overload and recommend scaling  

**KEY FEATURES:**  
- CPU usage  
- Memory usage  
- Network latency  
- Resource allocation limits  

**GOAL:** Bridge the gap between allocated vs actual resource usage.

---

## APPROACH

**1. DATA ACQUISITION**  
- Import datasets from Kaggle

**2. DATA CLEANING & PREPROCESSING**  
- Missing value handling  
- Normalization of numerical features  
- Encoding categorical variables  

**3. EXPLORATORY DATA ANALYSIS (EDA)**  
- Visualize CPU/memory trends  
- Identify correlations  
- Detect overload patterns  

**4. FEATURE ENGINEERING**  
- CPU-to-memory ratio  
- Utilization efficiency  
- Rolling averages  
- Network load  

**5. MODEL BUILDING**  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- K-Means (for clustering)

**6. MODEL EVALUATION**  
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

**7. INSIGHTS & RECOMMENDATIONS**  
- Key factors causing overload  
- Suggestions for optimized autoscaling  

**8. REPORT & VISUALIZATION**  
- Model comparison  
- Feature importance  
- Final summary  

---

## TIMELINE

**WEEK 1:** Data acquisition + preprocessing  
**WEEK 2:** EDA + feature engineering  
**WEEK 3:** Model building + evaluation  
**WEEK 4:** Insights, recommendations, final report  

---

## POSSIBLE ISSUES

- Accuracy may not generalize to real Kubernetes environments  
- Models rely on dataset quality; unseen spikes may be hard to predict  
- Some metrics may misalign with model needs  
- Missing or incomplete data reduces accuracy  
- Real-time implementation requires streaming pipelines  

---

