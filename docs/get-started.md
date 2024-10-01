# Network Intrusion Detection System (NIDS)

This page provides an overview of the **Network Intrusion Detection System (NIDS)** project, its objectives, and the machine learning models used to detect intrusions in IoT environments.

## Project Overview

The **NIDS project** focuses on enhancing the security of IoT networks using machine learning techniques to detect both benign and malicious traffic. The system leverages the **CIC IoT 2023 dataset**, which represents real-world network conditions, making it ideal for testing network intrusion detection models.

See [Project Objective](#project-objective) for more details.

---

## Project Objective

This project compares three machine learning models to evaluate their effectiveness in detecting network intrusions:

- **Logistic Regression**
- **Multilayer Perceptron (MLP)**
- **Hybrid Stacking Classifier**

Each model is tested against common attack types such as **DDoS-ICMP_Flood**, **DDoS-UDP_Flood**, and **Mirai** attacks.

See [Problem Description](#problem-description) for more details.

---

## Problem Description

As the adoption of **Internet of Things (IoT)** technology continues to grow across industries, it brings with it a host of new security vulnerabilities. The interconnected nature of IoT devices creates complex and dynamic network environments that are increasingly difficult to protect. Traditional network intrusion detection systems (NIDS) rely on **signature-based methods** that are ineffective at identifying emerging and sophisticated threats.

Intrusions such as **Distributed Denial of Service (DDoS)** attacks and botnets like **Mirai** exploit these vulnerabilities, leading to potentially devastating breaches. To counter these challenges, advanced **machine learning (ML)** techniques have emerged as promising tools for enhancing the detection capabilities of **Intrusion Detection Systems (IDS)**.

This research conducts a comparative analysis of three machine learning models:

- **Logistic Regression**
- **Multilayer Perceptron (MLP)**
- **Hybrid Stacking Classifier**

These models are trained and evaluated on the **CIC IoT 2023 dataset** to detect benign and malicious traffic, focusing on attacks such as **DDoS-ICMP_Flood**, **DDoS-UDP_Flood**, and **Mirai variants**.

---

## Literature Review

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is a crucial step in understanding the dataset's characteristics and underlying patterns before applying machine learning models. In this project, we analyze the **CIC IoT 2023 dataset** to uncover insights related to benign and malicious traffic.

### 1. Flow Duration

**Distribution**:

- Benign traffic typically exhibits low response times, with a histogram showing a spike at the origin and a long tail, indicating that most connections are transient.

**Relationships**:

- Certain protocols and packet structures correlate with flow duration, revealing patterns in how different types of traffic behave over time.

### 2. Header Length

**Distribution**:

- The benign traffic header length distribution shows a small mean value, indicating packets conform to a typical format, although some variance exists.

**Relationships**:

- The header lengths can vary across different protocol types and traffic rates, suggesting a diversity of packet structures depending on the protocols used.

### 3. Protocol Type

**Distribution**:

- KDE plots reveal several peaks in protocol types used in benign traffic, highlighting the use of multiple standard protocols.

**Relationships**:

- Some protocols generate higher traffic rates than others, indicating that specific types of protocols dominate the traffic patterns.

### 4. Rate and Source Rate

**Distribution**:

- Both the rate and source rate for benign traffic primarily show low values, with occasional higher rates during peak activity.

**Relationships**:

- There is a positive correlation between rate and source rate, as higher packet transmission rates typically correspond with increased source rates.

### 5. Benign Traffic Analysis

![VuePress Logo](/images/beningtraffic.jpg)

- Benign traffic displays versatility in protocol usage, characterized by short flow durations and normalized traffic density, indicating typical network behavior.

### 6. DDoS-ICMP_Flood Attack Analysis

**Protocol Distribution**:

- Interestingly, the attack traffic for DDoS-ICMP_Flood primarily relies on the ICMP protocol, which is characteristic of such attacks.

**Flow Duration**:

- The attack traffic shows higher total flow durations compared to benign traffic, with variability due to the nature of DDoS attacks.

**Header Length**:

- In attack traffic, header lengths may show more uniformity, particularly if configurations are adjusted to optimize attack effectiveness.

**Rates**:

- Overall attack traffic rates surpass benign traffic rates, reflecting the intention of the attack to overwhelm the target.

### Conclusion of EDA

The EDA reveals significant differences between benign traffic and various attack types, particularly in flow duration, header lengths, and traffic rates. Understanding these differences is crucial for enhancing intrusion detection systems, allowing for the implementation of more effective detection mechanisms to counteract malicious activities in IoT networks.

### A Novel Approach for Developing a Network Intrusion Detection System

Al-Garadi et al. (2021) propose a multipronged approach that enhances the accuracy of identifying intrusions while lowering the noise rate through the use of machine learning. The authors note some disadvantages of traditional NIDS, particularly their inability to process high-dimensional data and adapt to constantly changing threats.

The proposed system utilizes deep learning models:

- **Convolutional Neural Networks (CNN)**
- **Long Short-Term Memory (LSTM) Networks**

This approach involves:

- Data manipulation
- Data feature extraction
- Training of classifier models on the **CICIDS2018** and **Edge_IIoT** benchmark datasets.

When compared to pure CNN or LSTM architectures, the hybrid model significantly outperforms, achieving a detection rate of **99%**, whereas the traditional models on the CICIDS2018 dataset have the lowest detection rate of **64%**. This paper offers guidelines on integrating advanced machine learning in cybersecurity applications, especially in NIDS, and proposes new research directions for development.

### Machine Learning in Cybersecurity

A recent study by Jain, Waoo & Chauhan (2022) illustrates how the use of ML can significantly reduce cybersecurity threats. The study encompasses a wide range of ML algorithms, including:

- Decision Trees
- Support Vector Machines (SVM)
- Neural Networks

The authors evaluate the strengths and weaknesses of each IDS approach and emphasize the importance of:

- Feature selection
- Data pre-processing
- Model assessment

The study also addresses real-world challenges in ML-based IDS, such as:

- Handling high-dimensionality
- Addressing dataset skewness
- Model sensitivity to adversarial perturbations

Based on a review of articles published in the last five years, this study provides a comprehensive overview and explores future research avenues to enhance the performance and reliability of IDS in cybersecurity through ML.

### Autoencoder for Anomaly Detection

The use of autoencoder-based models in network security and anomaly detection was investigated by Xu et al. (2021), utilizing the **NSL-KDD** dataset as a benchmark for IDS evaluation.

Key findings include:

- A refined autoencoder approach that improves the detection of network anomalies through advanced data preprocessing and feature extraction.
- The autoencoder model is designed to learn the normal behavior of networks, enabling it to easily identify intrusion or anomaly behaviors.

Outcomes suggest that the proposed autoencoder model offers a higher and more reliable detection rate with fewer false positives compared to other models. This research provides both theoretical and empirical insights into network security, demonstrating that autoencoder algorithms can effectively identify complex and dynamic cyber threats.

### IoT Botnet Detection via Autoencoders

In the research article by Meidan et al. (2018) the authors discuss a method of detection of IoT botnet attacks employing deep autoencoders. The authors stress that despite the undeniable advantages of increasing the numbers of interconnected devices, botnets pose a considerable danger to the IoT environment, as the latter comprises numerous devices that can be subverted for unlawful purposes. The proposed detection system involves employing deep autoencoders in this process since they are capable of filtering network traffic in a way that enables them to identify traffic patterns typical of botnet. The model used to provide recommendations is trained on a vast amount of IoT network traffic data and the model’s efficacy, in particular, the ability to correctly detect causative traffic and the false positives generated by the model are assessed. Based on this study, it is possible to conclude that deep autoencoder model indeed allows for the detection of IoT botnet attacks with high accuracy, and can therefore contribute to the improvement of IoT security. In light of these findings, this study highlights the central role of strengthening existing methods to establish new approaches as an answer to the difficulties of IoT environments.

---

## Methodology

The project follows an experimental design using the **CIC IoT 2023 dataset**. The key steps include:

1. **Data Preprocessing**: Normalization, handling missing values, and applying **SMOTE** for class balancing.
2. **Model Training**: Three machine learning models were trained:
   - **Logistic Regression**
   - **Multilayer Perceptron (MLP)**
   - **Hybrid Stacking Classifier**

See [Model Descriptions](#model-descriptions) for details on each model.

---

## Model Descriptions

### Logistic Regression

A statistical model for binary classification. While simple and fast, it may produce more false positives compared to other models.

### Multilayer Perceptron (MLP)

```python
import os
import pandas as pd
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10, random_state=42, verbose=True)

with tqdm(total=mlp.max_iter, desc="Training MLP", unit="iteration") as pbar:
    def update_progress(*args):
        pbar.update()

    mlp._fit_stages = update_progress
    mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)

accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f'Accuracy of MLP on test set: {accuracy_mlp}')

```

A **feedforward neural network** capable of learning complex patterns in network traffic, providing improved detection accuracy over logistic regression.

### Hybrid Stacking Classifier

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('knn', KNeighborsClassifier(n_jobs=-1)),
    ('et', ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42))
]

final_estimator = LogisticRegression(n_jobs=-1)

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)

print("Training the stacking classifier...")
for name, clf in tqdm(estimators, desc="Training Base Estimators", total=len(estimators)):
    clf.fit(X_train, y_train_encoded)

stacking_clf.fit(X_train, y_train_encoded)

print("Making predictions on the test set...")
y_pred_stack = stacking_clf.predict(X_test)

accuracy_stack = accuracy_score(y_test_encoded, y_pred_stack)
print(f'Accuracy of Stacking Classifier: {accuracy_stack}')
```

An **ensemble model** that combines the strengths of various classifiers (e.g., RandomForest, GradientBoosting) to achieve superior performance.

See [Results](#results) for performance metrics.

---

## Results

### Logistic Regression

- **Accuracy**: 88.50%
- **Precision**: 84.34%
- **Recall**: 94.48%
- **F1 Score**: 89.13%

### Multilayer Perceptron (MLP)

- **Accuracy**: 92.81%
- **Precision**: 90.30%
- **Recall**: 95.87%
- **F1 Score**: 93.01%

### Hybrid Stacking Classifier

- **Accuracy**: 99.80%
- **Precision**: 99.93%
- **Recall**: 99.66%
- **F1 Score**: 99.80%

![VuePress Logo](/images/final.png)

The **Hybrid Stacking Classifier** outperforms the other models, achieving near-perfect results.

See [Conclusion](#conclusion) for the final insights.

---

## Conclusion

The **Hybrid Stacking Classifier** is the most effective model for detecting intrusions in IoT environments, with the highest accuracy and precision. Future work could explore additional datasets and expand the use of other machine learning models.
