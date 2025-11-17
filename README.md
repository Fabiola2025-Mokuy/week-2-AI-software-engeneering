# week-2-AI-software-engeneering
Readme
Com certeza! Aqui está a tradução completa do seu arquivo README para o inglês.
________________________________________
🚀 README: Potable Water Detection Program (ANN + KMeans)
This repository contains the code for a Machine Learning project developed in Google Colab to classify water potability and identify quality groups using supervised and unsupervised learning.
________________________________________
🎯 Project Goal
The main objective is to build an Artificial Intelligence model to support the UN Sustainable Development Goal (SDG) 6 (Clean Water and Sanitation) through:
1.	Classification (Supervised): Predicting whether a water sample is Potable (1) or Non-Potable (0) using an Artificial Neural Network (ANN).
2.	Clustering (Unsupervised): Automatically identifying water quality groups (clusters) to prioritize areas for intervention.
________________________________________
🛠️ Technologies Used
Tool	Usage
Python	Main programming language.
Jupyter Notebook / Google Colab	Development environment.
Pandas & NumPy	Data manipulation and simulation.
Scikit-learn	Data preprocessing (Cleaning, Normalization), Splitting, and KMeans Algorithm.
TensorFlow / Keras	Building, training, and evaluation of the Neural Network (ANN).
Matplotlib & Seaborn	Visualization of results and loss curves.
________________________________________
📊 Dataset Structure
The program is built based on the following chemical and physical water characteristic columns:
Column	Description
ph	Water's acidity/alkalinity level.
Hardness	Water's capacity to precipitate soap (hardness).
Solids	Total Dissolved Solids (TDS).
Chloramines	Chloramine level.
Sulfate	Sulfate level.
Conductivity	Electrical conductivity.
Organic_carbon	Total Organic Carbon level.
Trihalomethanes	Trihalomethanes level.
Turbidity	Measure of water clarity (turbidity).
Potability	Target variable: 1 (Potable) or 0 (Non-Potable).
________________________________________
🚀 How to Run the Program
1.	Open Colab: Open the code file in Google Colab.
2.	Replace Data: In Section 1, replace the data simulation block with my CSV file loading (df = pd.read_csv(water_potability.csv')).
3.	Execution: Run the cells sequentially.
________________________________________
📈 Process Steps (Pipeline)
1. Data Preprocessing (Scikit-learn)
•	Cleaning: Missing values (NaN) are filled using the column's median.
•	Splitting: Data is separated into Features (X) and Target (y).
•	Normalization: StandardScaler is applied to the features data to ensure the Neural Network converges efficiently.
•	Train/Test Split: Data is divided into 80% for training and 20% for testing.
2. Supervised Learning (Classification)
•	Model: Multi-layer Artificial Neural Network (ANN) (Dense) using TensorFlow/Keras.
•	Training: The model is trained for 50 epochs.
•	Evaluation: Accuracy and F1-Score metrics are used to measure performance, with the F1-Score being the critical indicator of balance and safety.
3. Unsupervised Learning (Clustering)
•	Model: KMeans Algorithm with $K=3$ (Three Clusters: Good, Medium, Bad).
•	Analysis: The goal is to group samples by their physical and chemical characteristics (without looking at the Potability label), identifying groups of pollution risk.
________________________________________
⚠️ Ethical Considerations
The project acknowledges that sampling bias (training the model only on clean water data) can lead to high-risk False Positives in underserved communities.
Ethical Commitment: Seeking geographic diversity and class balance is essential to ensure equity and safety in accessing water potability information, supporting SDG 6.

