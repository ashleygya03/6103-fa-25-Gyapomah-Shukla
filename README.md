## DATS 6103 DATA MINING - ADITI SHUKLA, ASHLEY GYAPOMAH

---

## OBJECTIVE  
**OBJECTIVE:** Predict upcoming container overloads in Kubernetes before autoscaling events occur.

---

## PROBLEM STATEMENT  
In Kubernetes clusters, containers can exceed CPU and memory limits *before* Kubernetes creates new pods.  
This delay in scaling results in:  
- Buffering  
- Slow response times  
- Temporary performance issues  

This project aims to predict overload **before** it happens, enabling proactive and timely scaling.

---

## SOLUTION  
Using the Kubernetes **resource allocation dataset**, we will:  
- Identify patterns that indicate CPU or memory overload  
- Engineer utilization features (CPU/Memory ratios, request ratios, overall load)  
- Train machine learning models to forecast when scaling is needed  
- Evaluate performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC  

This approach helps create a predictive autoscaling mechanism instead of relying solely on reactive scaling.

---
## DATASET OVERVIEW  
The project uses the **Kubernetes Resource Allocation Dataset** containing 15,000 records of pod-level resource usage.

### KEY FEATURES INCLUDED:
- **CPU Metrics:**  
  `cpu_request`, `cpu_limit`, `cpu_usage`
- **Memory Metrics:**  
  `memory_request`, `memory_limit`, `memory_usage`
- **Pod Metadata:**  
  `pod_name`, `namespace`, `pod_status`, `node_name`
- **Operational Metrics:**  
  `restart_count`, `uptime_seconds`, `deployment_strategy`, `scaling_policy`
- **Network Metric:**  
  `network_bandwidth_usage`

The dataset provides accurate insights into how pods consume resources, making it ideal for predicting overload events.

---

## DATA PREPROCESSING  - COMPLETED
To ensure reliable model performance, several preprocessing steps were applied:

- Standardized column names  
- Removed invalid or negative values  
- Verified no missing values in critical resource columns  
- Sorted records by `pod_name` and timestamp (where applicable)  
- Encoded categorical features using **OneHotEncoding**  
- Normalized numerical features using **StandardScaler**  
- Performed an 80/20 **train-test split** with stratification to preserve class balance

These steps ensured clean, consistent, and high-quality data for machine learning.

---

## FEATURE ENGINEERING  - COMPLETED
Additional meaningful metrics were derived to improve prediction performance:

- **CPU Utilization:** `cpu_usage / cpu_limit`  
- **Memory Utilization:** `memory_usage / memory_limit`  
- **CPU Request Ratio:** `cpu_usage / cpu_request`  
- **Memory Request Ratio:** `memory_usage / memory_request`  
- **Overall Load Score:** Average of CPU and Memory utilization  

These engineered features highlight stress patterns that correlate strongly with overload.

---

## TARGET VARIABLE  - COMPLETED 
The target label **need_new_pod** identifies when a pod is overloaded.

A pod is considered overloaded when:
- **CPU Utilization > 0.75**, or  
- **Memory Utilization > 0.75**

### CLASS DISTRIBUTION:
- **1 — Overloaded:** 83.7%  
- **0 — Not Overloaded:** 16.3%  

This imbalance required careful stratification during the train-test split to avoid biased training.

---

## MODELING APPROACH  - COMPLETED 
Several machine learning models were trained and evaluated using the same preprocessing pipeline:

- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Random Forest Classifier**  
- **Gradient Boosting Classifier**  
- **XGBoost Classifier** 

### EVALUATION METRICS: - COMPLETED 
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

These models help determine the most reliable method for predicting overload behavior.

---
## CONCLUSION  - COMPLETED 
This project successfully demonstrates how machine learning can:

- Predict pod overload *before* it happens  
- Reduce scaling delays  
- Improve system responsiveness  
- Support smarter, proactive autoscaling decisions  

By leveraging resource utilization metrics and engineered features, the model provides a high-accuracy solution for Kubernetes autoscaling prediction.

---

## FUTURE WORK  
Potential enhancements include:

- Integrating real-time latency and performance metrics  
- Building a multistep time-series forecasting model  
- Deploying the model within a Kubernetes cluster  
- Implementing live alerts for overload prediction  
- Testing with real production workloads

---
