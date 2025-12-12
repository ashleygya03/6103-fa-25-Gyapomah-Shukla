
# **PROJECT TITLE:-PREDICTING KUBERNATES POD OVERLOAD USING MACHINE LEARNING**
---
## **DATS-6103:- DATA MINING**

## **PROJECT BY:- ADITI SHUKLA, ASHLEY GYAPOMAH**

---
## **PROBLEM STATEMENT**
In large systems, some containers consume excessive CPU and memory resources. When a container reaches its usage limit (defined by a threshold), Kubernetes automatically creates a new container (pod) to balance the load. However, during this scaling process, there is a short delay before the new container becomes active, which can cause temporary performance issues such as buffering or slow response times for users.

---
## **SOLUTION**
Using the Kubernetes resource and performance metrics dataset, we aim to analyze when such lags or scaling events occur and develop machine learning models to  predict  when a new 
container will be required. By doing so, we can anticipate resource bottlenecks and trigger 
scaling actions proactively  effectively bridging the gap between high load and the delay in 
Kubernetes automatic scaling mechanism. 

---
## **IMPACT**
Instead of waiting for containers to exceed resource thresholds, the machine learning model 
anticipates potential overloads and triggers scaling actions early.

---
## **DATASET OVERVIEW**
The dataset consists of 15,000 Kubernetes pod-level observations with 15 columns detailing CPU and memory requests, limits, usage, pod metadata, uptime, restart counts, and scaling configurations. All values were complete with no missing entries, allowing for a clean modeling pipeline. Numeric columns captured resource behavior, while categorical columns (such as namespace, node name, and scaling policy) provided structural context. The dataset strongly reflects real world container orchestration scenarios where resource utilization patterns drive scaling decisions.

---
## **DATA CLEANING & PREPROCESSING**

The data cleaning process standardized column names, ensured correct data types, and removed invalid entries. Key numeric columns including CPU request, CPU limit, memory usage, and memory limit were validated to contain only non-negative values. Because the dataset had no missing values, no imputation was required. Beyond cleaning, preprocessing steps were performed to prepare the dataset for machine learning models:

### **ONE-HOT ENCODING (Categorical Feature Transformation)**

The dataset contained several categorical variables such as:
- namespace
- node_name
- pod_status
- deployment_strategy
- scaling_policy

Since machine learning algorithms cannot interpret categorical labels directly, One-Hot Encoding was applied. This transformed each categorical column into a set of binary indicator variables, enabling models to learn from categorical differences without imposing any ordinal relationship.
This step ensured compatibility with logistic regression, decision trees, ensemble methods, and gradient boosting algorithms.

### **STANDARD SCALING (Normalization of Numeric Features)**

Because the dataset contained numeric columns that differed significantly in scale (e.g., CPU usage values vs. uptime_seconds), StandardScaler was used to normalize all numeric features.

Standardization transforms each feature using:

$𝑧=𝑥−𝜇/𝜎$

This ensures:
- Models like Logistic Regression and Gradient Boosting converge faster
- Features contribute proportionally
- Scale-sensitive models operate more efficiently
- Tree-based models are scale invariant, but for consistency and fairness across models, scaling was applied uniformly.

### **TRAIN–TEST SPLIT**

The dataset was split into:
- Training Set: 12,000 rows (80%)
- Test Set: 3,000 rows (20%)

The split ensured:
- Unbiased model evaluation
- Prevention of data leakage
- Fair comparison between training and prediction performance

This separation is crucial for detecting overfitting but since our features were deterministic and strongly predictive, even the test results showed exceptionally high accuracy.

---

## **FEATURE ENGINEERING**
Feature engineering was the most critical step in this project because the raw dataset only contained absolute CPU and memory values, which do not directly indicate overload conditions. To accurately reflect Kubernetes autoscaling logic, we transformed these raw metrics into meaningful utilization-based features. CPU utilization and memory utilization were computed by dividing usage by the respective limits, capturing the proportion of resources consumed by each pod. Additionally, CPU request ratio and memory request ratio were derived by comparing actual usage against the resource requests guaranteed by Kubernetes, revealing whether a pod was bursting beyond its reserved allocation. An overall load feature was also created by averaging CPU and memory utilization, providing a combined indicator of system pressure. These engineered features directly corresponded to how the target variable was defined—based on utilization thresholds (greater than 0.75) which created a strong, deterministic relationship between the input features and the target. As a result, the machine learning models were able to learn the overload patterns with exceptional accuracy. The high performance is not a sign of overfitting but rather a reflection of how well the engineered features captured the underlying autoscaling behavior built into Kubernetes.

### **Why Feature Engineering Was Important?**

- Raw CPU/memory values alone cannot indicate overload because they don’t account for pod limits.
- Kubernetes autoscaling is based on utilization, not raw usage; engineering utilization features matched real system behavior.
- Engineered features created a direct connection between inputs and the target variable.

### **FEATURES AND FORMULA:**

- **CPU Utilization:**  
  $cpu\_utilization = \frac{cpu\_usage}{cpu\_limit + \epsilon}$  
  → Shows percentage of CPU capacity used.

- **Memory Utilization:**  
  $memory\_utilization = \frac{memory\_usage}{memory\_limit + \epsilon}$  
  → Identifies memory pressure and potential OOM risk.

- **CPU Request Ratio:**  
  $cpu\_request\_ratio = \frac{cpu\_usage}{cpu\_request + \epsilon}$  
  → Reveals whether the pod is using more than requested.

- **Memory Request Ratio:**  
  $memory\_request\_ratio = \frac{memory\_usage}{memory\_request + \epsilon}$  
  → Indicates memory bursts beyond reservation.

- **Overall Load:**  
  $overall\_load = \frac{cpu\_utilization + memory\_utilization}{2}$  
  → Combines CPU and memory pressure into a single metric.

### **What Actually Happened?**

- Feature engineering normalized resource behavior, allowing fair comparison across pods of different sizes. 
- Utilization features directly reflected how Kubernetes detects overload, making the target variable deterministic. 
- Models learned the overload boundary cleanly because engineered features perfectly aligned with the threshold (0.75). 
- This alignment produced very high accuracy without overfitting.

---
## **EXPLORATORY DATA ANALYSIS(EDA)**

### **CPU UTILIZATION DISTRIBUTION**

The CPU utilization histogram provides insight into how CPU resources are consumed across all pods. The distribution is heavily right-skewed, with most pods operating between 0 and 1.5 utilization, but with several pods reaching values as high as 7–8. A utilization value greater than 1.0 indicates that a pod is consuming more CPU than its allocated limit, signaling CPU saturation. This behavior is critical because CPU pressure is a major trigger for Kubernetes autoscaling. The shape of this distribution shows that overload is not a rare anomaly but a common behavior in this dataset, validating the importance of utilization-based engineered features. This plot confirms that CPU utilization will play a significant role in predicting whether a pod requires scaling and supports using engineered features to better model overload conditions.

### **MEMORY UTILIZATION DISTRIBUTION**

The memory utilization distribution shows an even stronger right-skew than CPU utilization, with many pods exhibiting memory consumption far beyond their memory limits, reaching values above 30× the allocated limit. Such extreme spikes may indicate memory intensive workloads, bursty data processing, or potential memory leaks. Memory overloads are especially important because they often cause Out-of-Memory (OOM) kills and force pod restarts. This distribution highlights that memory pressure is highly volatile and a strong determinant of overload conditions. From an ML perspective, this validates the inclusion of engineered features like memory_utilization and memory_request_ratio, which help the model detect memory driven scaling events far more effectively than raw memory values alone.

### **CPU VS MEMORY UTILIZATION SCATTERPLOT**

The scatterplot comparing CPU utilization to memory utilization reveals no strong linear relationship between the two resource types. Pods may experience high CPU usage with low memory usage, or vice versa. Memory usage shows significantly higher extremes than CPU, reaching over 30× memory limit, while CPU peaks around 8× CPU limit. The absence of a clear pattern shows that overload is multi dimensional, depending on both CPU and memory independently. This analysis justifies the creation of the overall_load feature, which averages CPU and memory utilization to represent total system pressure. It also supports the use of multi-feature machine learning models, rather than single variable heuristics.

### **CORRELATION HEATMAP**

The correlation heatmap visualizes relationships among all numeric features, including raw metrics and engineered features. As expected, raw usage metrics strongly correlate with their respective utilization metrics for example, cpu_usage correlates closely with cpu_utilization, and memory_usage correlates with memory_utilization. Overall_load shows strong correlation with memory utilization, suggesting that memory usage contributes more substantially to overall pressure in this dataset than CPU usage does. Non-resource features such as uptime_seconds, restart_count, and network_bandwidth_usage show weak correlations with overload-related variables, indicating they are less influential. This validation step confirms that our engineered features accurately express pod behavior and are aligned with predictive patterns required for modeling.

---
## **STATISTICAL TESTING**

Statistical tests were conducted to understand whether different variables show meaningful differences or relationships related to pod overload behavior. These tests help validate assumptions from EDA and support our interpretation of which features are significant predictors. Below are detailed explanations for each statistical test performed.

### **T-TEST: CPU USAGE VS OVERLOAD**

The two-sample independent t-test was used to compare CPU usage between overloaded pods (need_new_pod = 1) and non-overloaded pods (need_new_pod = 0). In our results, we obtained a T-statistic of 134.6196 and a p-value of 0.0, which is far below the standard significance threshold of 0.05. This indicates a highly significant difference in CPU usage between the two groups. Overloaded pods consistently show much higher CPU usage than non-overloaded pods. This test confirms what we observed in EDA: CPU usage is a strong driver of overload behavior. The statistical significance also validates the importance of engineered features such as cpu_utilization and cpu_request_ratio, which directly capture the relationship between CPU pressure and the overload target variable.

### **ANOVA: CPU USAGE ACROSS NAMESPACES**

A one-way ANOVA test was performed to compare CPU usage across multiple namespaces (e.g., default, kube-system, monitoring). The F-statistic was 0.7011 with a p-value of 0.55127, well above the significance threshold. This means there is no statistically significant difference in CPU usage between namespaces. In other words, workload intensity is not dependent on namespace labels—CPU behavior is fairly consistent across namespaces. This finding is important because it confirms that namespace is not a meaningful predictor of overload and should not be weighted heavily in the model. It also justifies why CPU-related features (not namespace metadata) dominate the modeling process.

### **CHI-SQUARE TEST: SCALING POLICY VS OVERLOAD**

The chi-square test was used to evaluate whether scaling policies (categorical variable) are associated with overload behavior. The test produced a Chi-square value of 0.6446 and a p-value of 0.4220, showing no significant relationship. This means that the choice of scaling policy (manual, auto, etc.) does not statistically influence whether a pod becomes overloaded. This is an important insight: it highlights that overload is primarily driven by real time resource usage, not by configuration metadata. Even if different pods have different scaling strategies, their likelihood of overload is determined by CPU and memory pressure, confirming the strength of utilization-based engineered features.

---
## **DEFINING TARGET VARIABLE**

The target variable need_new_pod was created by defining an overload condition: a pod is labeled as requiring a new pod if either CPU utilization or memory utilization exceeds 0.75. This threshold-based logic mirrors real autoscaling strategies where resource pressure triggers horizontal scaling actions. Because our engineered features directly captured utilization levels, the target variable became closely aligned with patterns in the engineered data. This deterministic relationship is a major factor behind the strong model performance, as the inputs were highly predictive of the output by design.

---
## **MODELING AND EVALUATION**

### **LOGISTIC REGRESSION**

Logistic Regression performed well overall, achieving 93.5% accuracy, 95.9% precision, 96.2% recall, and an F1-score of 0.9612. While this model was able to capture the general relationship between resource utilization and overload, it performed noticeably worse than the tree based models. The classification report shows that Logistic Regression struggled more with the minority class (non-overloaded pods), achieving only 80% F1-score for class 0, as the linear decision boundary is unable to fully capture the non-linear and threshold based relationships present in the data. The ROC-AUC of 0.9787 still indicates strong separability, meaning Logistic Regression does learn meaningful patterns, but its limitations appear when handling complex interactions between CPU and memory utilization. This model serves as a good baseline but is not ideal for systems where overload dynamics are inherently non-linear.

### **DECISION TREE**

The Decision Tree model achieved exceptional performance, with 99.93% accuracy, 0.9996 precision, 0.9996 recall, and an F1-score of 0.9996. These near-perfect metrics reflect the deterministic nature of the overload classification, where clear thresholds (e.g., utilization > 0.75) allow the tree to create precise decision boundaries. Decision Trees naturally excel in such rule based environments, where engineered features such as CPU and memory utilization directly match the structure of logical comparisons. Additionally, the Decision Tree successfully distinguished both classes with almost perfect accuracy, demonstrating that the patterns in the data align extremely well with the binary overload logic. The ROC-AUC score of 0.9988 further confirms that this model is highly effective and captures the underlying structure of the data with minimal error.


### **RANDOM FOREST**

The Random Forest model also delivered extremely strong performance, achieving 99.5% accuracy, 0.9941 precision, 1.0 recall, and an F1-score of 0.9970. Random Forests improve upon Decision Trees by aggregating multiple trees, making the model more robust and less sensitive to noise or slight variations in split points. The perfect recall score indicates that the model identifies every overloaded pod, which is valuable in real-world autoscaling scenarios where missing an overload event can lead to performance degradation. The slight dip in precision compared to other models suggests a few false positives, but overall performance remains outstanding. The ROC-AUC score of 0.9999 highlights near perfect class separation. Random Forests excel here because the engineered utilization features provide clean, interpretable boundaries that ensembles can easily leverage.

### **GRADIENT BOOSTING**

Gradient Boosting emerged as the best-performing model, achieving a perfect 1.0000 score in accuracy, precision, recall, F1-score, and ROC-AUC. This flawless performance reflects the model’s ability to learn complex decision boundaries by iteratively correcting errors from previous weak learners. In this dataset, where overload classification is almost deterministic due to engineered features, Gradient Boosting effectively captures every detail of the decision surface without overfitting. Its sequential learning approach makes it highly sensitive to subtle interactions between CPU utilization, memory utilization, and overall load. The perfect ROC-AUC value confirms that the model distinguishes the two classes without any ambiguity. This result is expected when the feature target relationship is extremely strong and the engineered features closely mirror the logic used to define the target variable.

### **XGBOOST**

XGBoost also delivered outstanding results, with 99.93% accuracy, 0.9992 precision, 1.0 recall, and an F1-score of 0.9996, making it one of the top-performing models in the evaluation. The extremely high recall demonstrates that XGBoost identifies every overloaded pod, while strong precision indicates minimal false alarms. XGBoost’s gradient boosting framework, combined with regularization, makes it extremely powerful for structured datasets like this one. The ROC-AUC score of 0.999997 shows near-perfect separability. XGBoost performed almost identically to Gradient Boosting, but slightly below perfection due to very small prediction deviations on borderline samples. Overall, XGBoost is an excellent model for this task and is highly reliable for autoscaling prediction use cases.

### **MODEL COMPARISON(F1 SCORE & ROC-AUC)**

The model comparison visualizations clearly highlight how each classifier performs in terms of F1-Score and ROC-AUC, both of which are essential for evaluating classification performance in an imbalanced dataset like ours. The F1-score plot shows that Decision Tree, Random Forest, Gradient Boosting, and XGBoost all achieve exceptionally high F1 values close to 1.0 indicating almost perfect precision recall balance. Logistic Regression, while still performing strongly with an F1 of ~0.96, shows slightly lower performance compared to tree based models, likely because it assumes linear decision boundaries, whereas our engineered features interact non-linearly. The ROC-AUC comparison further strengthens this observation: tree-based models achieve ROC-AUC scores near 1.0, demonstrating exceptional ability to distinguish between overloaded vs. non-overloaded pods. Logistic Regression again shows a respectable ~0.98 ROC-AUC but remains the lowest among the models. The consistency of high scores across most models confirms that the target variable is highly deterministic, and the engineered features (CPU utilization, memory utilization, request ratios, and overall load) provide very strong signals for classification. This is why all models achieve extremely high performance not due to overfitting, but because the relationship between features and overload is inherently very strong and predictable.


---
## **CONCLUSION**
This project successfully demonstrates how engineered utilization based features can precisely model autoscaling behavior in Kubernetes environments. The target variable, defined by explicit CPU and memory thresholds, is highly deterministic, meaning that the relationship between features and labels is strong and direct. As a result, machine learning models especially tree-based ensembles achieved exceptionally high accuracy without overfitting. The project's results confirm that when feature engineering accurately captures domain logic, ML models learn effectively and produce reliable, interpretable predictions. This approach can be extended to real world scaling systems to improve performance, stability, and resource efficiency.
