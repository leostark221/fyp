<template><div><h1 id="network-intrusion-detection-system-nids" tabindex="-1"><a class="header-anchor" href="#network-intrusion-detection-system-nids"><span>Network Intrusion Detection System (NIDS)</span></a></h1>
<p>This page provides an overview of the <strong>Network Intrusion Detection System (NIDS)</strong> project, its objectives, and the machine learning models used to detect intrusions in IoT environments.</p>
<h2 id="project-overview" tabindex="-1"><a class="header-anchor" href="#project-overview"><span>Project Overview</span></a></h2>
<p>The <strong>NIDS project</strong> focuses on enhancing the security of IoT networks using machine learning techniques to detect both benign and malicious traffic. The system leverages the <strong>CIC IoT 2023 dataset</strong>, which represents real-world network conditions, making it ideal for testing network intrusion detection models.</p>
<p>See <a href="#project-objective">Project Objective</a> for more details.</p>
<hr>
<h2 id="project-objective" tabindex="-1"><a class="header-anchor" href="#project-objective"><span>Project Objective</span></a></h2>
<p>This project compares three machine learning models to evaluate their effectiveness in detecting network intrusions:</p>
<ul>
<li><strong>Logistic Regression</strong></li>
<li><strong>Multilayer Perceptron (MLP)</strong></li>
<li><strong>Hybrid Stacking Classifier</strong></li>
</ul>
<p>Each model is tested against common attack types such as <strong>DDoS-ICMP_Flood</strong>, <strong>DDoS-UDP_Flood</strong>, and <strong>Mirai</strong> attacks.</p>
<p>See <a href="#problem-description">Problem Description</a> for more details.</p>
<hr>
<h2 id="problem-description" tabindex="-1"><a class="header-anchor" href="#problem-description"><span>Problem Description</span></a></h2>
<p>As the adoption of <strong>Internet of Things (IoT)</strong> technology continues to grow across industries, it brings with it a host of new security vulnerabilities. The interconnected nature of IoT devices creates complex and dynamic network environments that are increasingly difficult to protect. Traditional network intrusion detection systems (NIDS) rely on <strong>signature-based methods</strong> that are ineffective at identifying emerging and sophisticated threats.</p>
<p>Intrusions such as <strong>Distributed Denial of Service (DDoS)</strong> attacks and botnets like <strong>Mirai</strong> exploit these vulnerabilities, leading to potentially devastating breaches. To counter these challenges, advanced <strong>machine learning (ML)</strong> techniques have emerged as promising tools for enhancing the detection capabilities of <strong>Intrusion Detection Systems (IDS)</strong>.</p>
<p>This research conducts a comparative analysis of three machine learning models:</p>
<ul>
<li><strong>Logistic Regression</strong></li>
<li><strong>Multilayer Perceptron (MLP)</strong></li>
<li><strong>Hybrid Stacking Classifier</strong></li>
</ul>
<p>These models are trained and evaluated on the <strong>CIC IoT 2023 dataset</strong> to detect benign and malicious traffic, focusing on attacks such as <strong>DDoS-ICMP_Flood</strong>, <strong>DDoS-UDP_Flood</strong>, and <strong>Mirai variants</strong>.</p>
<hr>
<h2 id="literature-review" tabindex="-1"><a class="header-anchor" href="#literature-review"><span>Literature Review</span></a></h2>
<h2 id="exploratory-data-analysis-eda" tabindex="-1"><a class="header-anchor" href="#exploratory-data-analysis-eda"><span>Exploratory Data Analysis (EDA)</span></a></h2>
<p>Exploratory Data Analysis (EDA) is a crucial step in understanding the dataset's characteristics and underlying patterns before applying machine learning models. In this project, we analyze the <strong>CIC IoT 2023 dataset</strong> to uncover insights related to benign and malicious traffic.</p>
<h3 id="_1-flow-duration" tabindex="-1"><a class="header-anchor" href="#_1-flow-duration"><span>1. Flow Duration</span></a></h3>
<p><strong>Distribution</strong>:</p>
<ul>
<li>Benign traffic typically exhibits low response times, with a histogram showing a spike at the origin and a long tail, indicating that most connections are transient.</li>
</ul>
<p><strong>Relationships</strong>:</p>
<ul>
<li>Certain protocols and packet structures correlate with flow duration, revealing patterns in how different types of traffic behave over time.</li>
</ul>
<h3 id="_2-header-length" tabindex="-1"><a class="header-anchor" href="#_2-header-length"><span>2. Header Length</span></a></h3>
<p><strong>Distribution</strong>:</p>
<ul>
<li>The benign traffic header length distribution shows a small mean value, indicating packets conform to a typical format, although some variance exists.</li>
</ul>
<p><strong>Relationships</strong>:</p>
<ul>
<li>The header lengths can vary across different protocol types and traffic rates, suggesting a diversity of packet structures depending on the protocols used.</li>
</ul>
<h3 id="_3-protocol-type" tabindex="-1"><a class="header-anchor" href="#_3-protocol-type"><span>3. Protocol Type</span></a></h3>
<p><strong>Distribution</strong>:</p>
<ul>
<li>KDE plots reveal several peaks in protocol types used in benign traffic, highlighting the use of multiple standard protocols.</li>
</ul>
<p><strong>Relationships</strong>:</p>
<ul>
<li>Some protocols generate higher traffic rates than others, indicating that specific types of protocols dominate the traffic patterns.</li>
</ul>
<h3 id="_4-rate-and-source-rate" tabindex="-1"><a class="header-anchor" href="#_4-rate-and-source-rate"><span>4. Rate and Source Rate</span></a></h3>
<p><strong>Distribution</strong>:</p>
<ul>
<li>Both the rate and source rate for benign traffic primarily show low values, with occasional higher rates during peak activity.</li>
</ul>
<p><strong>Relationships</strong>:</p>
<ul>
<li>There is a positive correlation between rate and source rate, as higher packet transmission rates typically correspond with increased source rates.</li>
</ul>
<h3 id="_5-benign-traffic-analysis" tabindex="-1"><a class="header-anchor" href="#_5-benign-traffic-analysis"><span>5. Benign Traffic Analysis</span></a></h3>
<p><img src="/images/beningtraffic.jpg" alt="VuePress Logo"></p>
<ul>
<li>Benign traffic displays versatility in protocol usage, characterized by short flow durations and normalized traffic density, indicating typical network behavior.</li>
</ul>
<h3 id="_6-ddos-icmp-flood-attack-analysis" tabindex="-1"><a class="header-anchor" href="#_6-ddos-icmp-flood-attack-analysis"><span>6. DDoS-ICMP_Flood Attack Analysis</span></a></h3>
<p><strong>Protocol Distribution</strong>:</p>
<ul>
<li>Interestingly, the attack traffic for DDoS-ICMP_Flood primarily relies on the ICMP protocol, which is characteristic of such attacks.</li>
</ul>
<p><strong>Flow Duration</strong>:</p>
<ul>
<li>The attack traffic shows higher total flow durations compared to benign traffic, with variability due to the nature of DDoS attacks.</li>
</ul>
<p><strong>Header Length</strong>:</p>
<ul>
<li>In attack traffic, header lengths may show more uniformity, particularly if configurations are adjusted to optimize attack effectiveness.</li>
</ul>
<p><strong>Rates</strong>:</p>
<ul>
<li>Overall attack traffic rates surpass benign traffic rates, reflecting the intention of the attack to overwhelm the target.</li>
</ul>
<h3 id="conclusion-of-eda" tabindex="-1"><a class="header-anchor" href="#conclusion-of-eda"><span>Conclusion of EDA</span></a></h3>
<p>The EDA reveals significant differences between benign traffic and various attack types, particularly in flow duration, header lengths, and traffic rates. Understanding these differences is crucial for enhancing intrusion detection systems, allowing for the implementation of more effective detection mechanisms to counteract malicious activities in IoT networks.</p>
<h3 id="a-novel-approach-for-developing-a-network-intrusion-detection-system" tabindex="-1"><a class="header-anchor" href="#a-novel-approach-for-developing-a-network-intrusion-detection-system"><span>A Novel Approach for Developing a Network Intrusion Detection System</span></a></h3>
<p>Al-Garadi et al. (2021) propose a multipronged approach that enhances the accuracy of identifying intrusions while lowering the noise rate through the use of machine learning. The authors note some disadvantages of traditional NIDS, particularly their inability to process high-dimensional data and adapt to constantly changing threats.</p>
<p>The proposed system utilizes deep learning models:</p>
<ul>
<li><strong>Convolutional Neural Networks (CNN)</strong></li>
<li><strong>Long Short-Term Memory (LSTM) Networks</strong></li>
</ul>
<p>This approach involves:</p>
<ul>
<li>Data manipulation</li>
<li>Data feature extraction</li>
<li>Training of classifier models on the <strong>CICIDS2018</strong> and <strong>Edge_IIoT</strong> benchmark datasets.</li>
</ul>
<p>When compared to pure CNN or LSTM architectures, the hybrid model significantly outperforms, achieving a detection rate of <strong>99%</strong>, whereas the traditional models on the CICIDS2018 dataset have the lowest detection rate of <strong>64%</strong>. This paper offers guidelines on integrating advanced machine learning in cybersecurity applications, especially in NIDS, and proposes new research directions for development.</p>
<h3 id="machine-learning-in-cybersecurity" tabindex="-1"><a class="header-anchor" href="#machine-learning-in-cybersecurity"><span>Machine Learning in Cybersecurity</span></a></h3>
<p>A recent study by Jain, Waoo &amp; Chauhan (2022) illustrates how the use of ML can significantly reduce cybersecurity threats. The study encompasses a wide range of ML algorithms, including:</p>
<ul>
<li>Decision Trees</li>
<li>Support Vector Machines (SVM)</li>
<li>Neural Networks</li>
</ul>
<p>The authors evaluate the strengths and weaknesses of each IDS approach and emphasize the importance of:</p>
<ul>
<li>Feature selection</li>
<li>Data pre-processing</li>
<li>Model assessment</li>
</ul>
<p>The study also addresses real-world challenges in ML-based IDS, such as:</p>
<ul>
<li>Handling high-dimensionality</li>
<li>Addressing dataset skewness</li>
<li>Model sensitivity to adversarial perturbations</li>
</ul>
<p>Based on a review of articles published in the last five years, this study provides a comprehensive overview and explores future research avenues to enhance the performance and reliability of IDS in cybersecurity through ML.</p>
<h3 id="autoencoder-for-anomaly-detection" tabindex="-1"><a class="header-anchor" href="#autoencoder-for-anomaly-detection"><span>Autoencoder for Anomaly Detection</span></a></h3>
<p>The use of autoencoder-based models in network security and anomaly detection was investigated by Xu et al. (2021), utilizing the <strong>NSL-KDD</strong> dataset as a benchmark for IDS evaluation.</p>
<p>Key findings include:</p>
<ul>
<li>A refined autoencoder approach that improves the detection of network anomalies through advanced data preprocessing and feature extraction.</li>
<li>The autoencoder model is designed to learn the normal behavior of networks, enabling it to easily identify intrusion or anomaly behaviors.</li>
</ul>
<p>Outcomes suggest that the proposed autoencoder model offers a higher and more reliable detection rate with fewer false positives compared to other models. This research provides both theoretical and empirical insights into network security, demonstrating that autoencoder algorithms can effectively identify complex and dynamic cyber threats.</p>
<h3 id="iot-botnet-detection-via-autoencoders" tabindex="-1"><a class="header-anchor" href="#iot-botnet-detection-via-autoencoders"><span>IoT Botnet Detection via Autoencoders</span></a></h3>
<p>In the research article by Meidan et al. (2018) the authors discuss a method of detection of IoT botnet attacks employing deep autoencoders. The authors stress that despite the undeniable advantages of increasing the numbers of interconnected devices, botnets pose a considerable danger to the IoT environment, as the latter comprises numerous devices that can be subverted for unlawful purposes. The proposed detection system involves employing deep autoencoders in this process since they are capable of filtering network traffic in a way that enables them to identify traffic patterns typical of botnet. The model used to provide recommendations is trained on a vast amount of IoT network traffic data and the model’s efficacy, in particular, the ability to correctly detect causative traffic and the false positives generated by the model are assessed. Based on this study, it is possible to conclude that deep autoencoder model indeed allows for the detection of IoT botnet attacks with high accuracy, and can therefore contribute to the improvement of IoT security. In light of these findings, this study highlights the central role of strengthening existing methods to establish new approaches as an answer to the difficulties of IoT environments.</p>
<hr>
<h2 id="methodology" tabindex="-1"><a class="header-anchor" href="#methodology"><span>Methodology</span></a></h2>
<p>The project follows an experimental design using the <strong>CIC IoT 2023 dataset</strong>. The key steps include:</p>
<ol>
<li><strong>Data Preprocessing</strong>: Normalization, handling missing values, and applying <strong>SMOTE</strong> for class balancing.</li>
<li><strong>Model Training</strong>: Three machine learning models were trained:
<ul>
<li><strong>Logistic Regression</strong></li>
<li><strong>Multilayer Perceptron (MLP)</strong></li>
<li><strong>Hybrid Stacking Classifier</strong></li>
</ul>
</li>
</ol>
<p>See <a href="#model-descriptions">Model Descriptions</a> for details on each model.</p>
<hr>
<h2 id="model-descriptions" tabindex="-1"><a class="header-anchor" href="#model-descriptions"><span>Model Descriptions</span></a></h2>
<h3 id="logistic-regression" tabindex="-1"><a class="header-anchor" href="#logistic-regression"><span>Logistic Regression</span></a></h3>
<p>A statistical model for binary classification. While simple and fast, it may produce more false positives compared to other models.</p>
<h3 id="multilayer-perceptron-mlp" tabindex="-1"><a class="header-anchor" href="#multilayer-perceptron-mlp"><span>Multilayer Perceptron (MLP)</span></a></h3>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py" data-title="py"><pre v-pre><code><span class="line"><span class="token keyword">import</span> os</span>
<span class="line"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd</span>
<span class="line"><span class="token keyword">from</span> tqdm <span class="token keyword">import</span> tqdm</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>neural_network <span class="token keyword">import</span> MLPClassifier</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>metrics <span class="token keyword">import</span> accuracy_score</span>
<span class="line"></span>
<span class="line">mlp <span class="token operator">=</span> MLPClassifier<span class="token punctuation">(</span>hidden_layer_sizes<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">100</span><span class="token punctuation">,</span> <span class="token number">50</span><span class="token punctuation">)</span><span class="token punctuation">,</span> max_iter<span class="token operator">=</span><span class="token number">10</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">,</span> verbose<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">with</span> tqdm<span class="token punctuation">(</span>total<span class="token operator">=</span>mlp<span class="token punctuation">.</span>max_iter<span class="token punctuation">,</span> desc<span class="token operator">=</span><span class="token string">"Training MLP"</span><span class="token punctuation">,</span> unit<span class="token operator">=</span><span class="token string">"iteration"</span><span class="token punctuation">)</span> <span class="token keyword">as</span> pbar<span class="token punctuation">:</span></span>
<span class="line">    <span class="token keyword">def</span> <span class="token function">update_progress</span><span class="token punctuation">(</span><span class="token operator">*</span>args<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">        pbar<span class="token punctuation">.</span>update<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">    mlp<span class="token punctuation">.</span>_fit_stages <span class="token operator">=</span> update_progress</span>
<span class="line">    mlp<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train<span class="token punctuation">,</span> y_train<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">y_pred_mlp <span class="token operator">=</span> mlp<span class="token punctuation">.</span>predict<span class="token punctuation">(</span>X_test<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">accuracy_mlp <span class="token operator">=</span> accuracy_score<span class="token punctuation">(</span>y_test<span class="token punctuation">,</span> y_pred_mlp<span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f'Accuracy of MLP on test set: </span><span class="token interpolation"><span class="token punctuation">{</span>accuracy_mlp<span class="token punctuation">}</span></span><span class="token string">'</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>A <strong>feedforward neural network</strong> capable of learning complex patterns in network traffic, providing improved detection accuracy over logistic regression.</p>
<h3 id="hybrid-stacking-classifier" tabindex="-1"><a class="header-anchor" href="#hybrid-stacking-classifier"><span>Hybrid Stacking Classifier</span></a></h3>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py" data-title="py"><pre v-pre><code><span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>preprocessing <span class="token keyword">import</span> LabelEncoder</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>ensemble <span class="token keyword">import</span> StackingClassifier<span class="token punctuation">,</span> RandomForestClassifier<span class="token punctuation">,</span> GradientBoostingClassifier<span class="token punctuation">,</span> ExtraTreesClassifier<span class="token punctuation">,</span> AdaBoostClassifier</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>neighbors <span class="token keyword">import</span> KNeighborsClassifier</span>
<span class="line"><span class="token keyword">from</span> xgboost <span class="token keyword">import</span> XGBClassifier</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>linear_model <span class="token keyword">import</span> LogisticRegression</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>metrics <span class="token keyword">import</span> accuracy_score</span>
<span class="line"><span class="token keyword">from</span> tqdm <span class="token keyword">import</span> tqdm</span>
<span class="line"></span>
<span class="line">label_encoder <span class="token operator">=</span> LabelEncoder<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">y_train_encoded <span class="token operator">=</span> label_encoder<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>y_train<span class="token punctuation">)</span></span>
<span class="line">y_test_encoded <span class="token operator">=</span> label_encoder<span class="token punctuation">.</span>transform<span class="token punctuation">(</span>y_test<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">estimators <span class="token operator">=</span> <span class="token punctuation">[</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'rf'</span><span class="token punctuation">,</span> RandomForestClassifier<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> n_jobs<span class="token operator">=</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'gb'</span><span class="token punctuation">,</span> GradientBoostingClassifier<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'knn'</span><span class="token punctuation">,</span> KNeighborsClassifier<span class="token punctuation">(</span>n_jobs<span class="token operator">=</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'et'</span><span class="token punctuation">,</span> ExtraTreesClassifier<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> n_jobs<span class="token operator">=</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'ada'</span><span class="token punctuation">,</span> AdaBoostClassifier<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">50</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'xgb'</span><span class="token punctuation">,</span> XGBClassifier<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line">final_estimator <span class="token operator">=</span> LogisticRegression<span class="token punctuation">(</span>n_jobs<span class="token operator">=</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">stacking_clf <span class="token operator">=</span> StackingClassifier<span class="token punctuation">(</span>estimators<span class="token operator">=</span>estimators<span class="token punctuation">,</span> final_estimator<span class="token operator">=</span>final_estimator<span class="token punctuation">,</span> n_jobs<span class="token operator">=</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Training the stacking classifier..."</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">for</span> name<span class="token punctuation">,</span> clf <span class="token keyword">in</span> tqdm<span class="token punctuation">(</span>estimators<span class="token punctuation">,</span> desc<span class="token operator">=</span><span class="token string">"Training Base Estimators"</span><span class="token punctuation">,</span> total<span class="token operator">=</span><span class="token builtin">len</span><span class="token punctuation">(</span>estimators<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    clf<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train<span class="token punctuation">,</span> y_train_encoded<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">stacking_clf<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train<span class="token punctuation">,</span> y_train_encoded<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Making predictions on the test set..."</span><span class="token punctuation">)</span></span>
<span class="line">y_pred_stack <span class="token operator">=</span> stacking_clf<span class="token punctuation">.</span>predict<span class="token punctuation">(</span>X_test<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">accuracy_stack <span class="token operator">=</span> accuracy_score<span class="token punctuation">(</span>y_test_encoded<span class="token punctuation">,</span> y_pred_stack<span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f'Accuracy of Stacking Classifier: </span><span class="token interpolation"><span class="token punctuation">{</span>accuracy_stack<span class="token punctuation">}</span></span><span class="token string">'</span></span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>An <strong>ensemble model</strong> that combines the strengths of various classifiers (e.g., RandomForest, GradientBoosting) to achieve superior performance.</p>
<p>See <a href="#results">Results</a> for performance metrics.</p>
<hr>
<h2 id="results" tabindex="-1"><a class="header-anchor" href="#results"><span>Results</span></a></h2>
<h3 id="logistic-regression-1" tabindex="-1"><a class="header-anchor" href="#logistic-regression-1"><span>Logistic Regression</span></a></h3>
<ul>
<li><strong>Accuracy</strong>: 88.50%</li>
<li><strong>Precision</strong>: 84.34%</li>
<li><strong>Recall</strong>: 94.48%</li>
<li><strong>F1 Score</strong>: 89.13%</li>
</ul>
<h3 id="multilayer-perceptron-mlp-1" tabindex="-1"><a class="header-anchor" href="#multilayer-perceptron-mlp-1"><span>Multilayer Perceptron (MLP)</span></a></h3>
<ul>
<li><strong>Accuracy</strong>: 92.81%</li>
<li><strong>Precision</strong>: 90.30%</li>
<li><strong>Recall</strong>: 95.87%</li>
<li><strong>F1 Score</strong>: 93.01%</li>
</ul>
<h3 id="hybrid-stacking-classifier-1" tabindex="-1"><a class="header-anchor" href="#hybrid-stacking-classifier-1"><span>Hybrid Stacking Classifier</span></a></h3>
<ul>
<li><strong>Accuracy</strong>: 99.80%</li>
<li><strong>Precision</strong>: 99.93%</li>
<li><strong>Recall</strong>: 99.66%</li>
<li><strong>F1 Score</strong>: 99.80%</li>
</ul>
<p><img src="/images/final.png" alt="VuePress Logo"></p>
<p>The <strong>Hybrid Stacking Classifier</strong> outperforms the other models, achieving near-perfect results.</p>
<p>See <a href="#conclusion">Conclusion</a> for the final insights.</p>
<hr>
<h2 id="conclusion" tabindex="-1"><a class="header-anchor" href="#conclusion"><span>Conclusion</span></a></h2>
<p>The <strong>Hybrid Stacking Classifier</strong> is the most effective model for detecting intrusions in IoT environments, with the highest accuracy and precision. Future work could explore additional datasets and expand the use of other machine learning models.</p>
</div></template>


