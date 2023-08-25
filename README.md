# ddos_attack_prediction_system
The DDoS Attack Prediction System is a Python-based project that incorporates a graphical
user interface (GUI) using the tkinter library. The system offers functionality for both
classification and anomaly detection tasks, employing a variety of machine learning
techniques. The user just needs a dataset pertaining to the case like the sample ones 
provided and then he/she can start their analysis.
Let's delve into a more detailed explanation of the project components and their
functionalities.
# 1. Classification Techniques:
The system provides the following classification techniques, each assigned with a specific
number:
# a. Random Forest:
Random Forest is an ensemble learning algorithm that constructs multiple decision trees
during training and predicts the class based on the mode of the predictions from individual
trees. For Random Forest, the system performs analysis to check if the dataset has a balanced
class distribution. If the class distribution is imbalanced, it suggests considering class
balancing or alternative techniques.
# b. Logistic Regression:
Logistic Regression is a statistical model used to predict the probability of a binary response
variable based on one or more predictor variables. To assess Logistic Regression, the system
analyses the dataset for linear separability by checking the correlation between variables and
the target variable. If strong correlations are found, it indicates potential linear separability.
# c. SVM (Support Vector Machines):
Support Vector Machines is a supervised learning algorithm that aims to find a hyperplane in
the feature space to maximize the separation between different classes. The system performs
an analysis specific to SVM by examining if the dataset has clearly separable classes. It
checks the class distribution and suggests feature engineering or alternative techniques if
clear separation is not observed.
# d. KNN (K-Nearest Neighbours):
K-Nearest Neighbours is a non-parametric algorithm used for both classification and
regression tasks. It assigns the class or value based on the majority vote or average of the k
nearest neighbours in the feature space. The system conducts an analysis to determine if the
dataset has a sufficient number of relevant features for accurate predictions.
# e. Gradient Boosting:
Gradient Boosting is an ensemble learning technique that combines weak models, typically
decision trees, sequentially to create a robust predictive model. The system performs an
analysis specific to Gradient Boosting by examining the presence of outliers that may affect
the model's performance. If outliers are present, it suggests employing outlier detection
techniques or alternative methods.
# 2. Anomaly Detection Techniques:
The system offers the following anomaly detection techniques, each associated with a distinct
number:
# a. Isolation Forest:
Isolation Forest is an unsupervised learning algorithm that isolates anomalies by randomly
selecting a feature and then selecting a split value within the feature's range. The system
analyses the dataset to check if it has a wide range of feature values, which can enhance the
Isolation Forest model's performance.
# b. Local Outlier Factor (LOF):
Local Outlier Factor is an unsupervised learning algorithm that assesses the abnormality of
data points based on the local density deviation of their neighbours. The system performs an
analysis specific to LOF by examining if the dataset exhibits distinct clusters, which can aid
in improving the LOF model's performance.
# c. One-Class SVM (Support Vector Machines):
One-Class SVM is an unsupervised learning algorithm that identifies outliers by separating
data points from the origin in a high-dimensional feature space. The system analyses the
dataset to determine if it has a clear separation between normal and anomalous instances,
which is favourable for the One-Class SVM model.
# d. DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
DBSCAN is a density-based clustering algorithm that groups data points based on their
proximity and density while labelling outliers as noise. The system performs an analysis
specific to DBSCAN by checking if the dataset exhibits an appropriate density for clustering,
which is suitable for the DBSCAN model.
# Dataset Selection and Preprocessing:
The system allows users to select a dataset in CSV format using a browse button. Once the
dataset is chosen, its file path is displayed on the interface. The selected dataset is then loaded
using the pandas library for further processing. Missing values are removed by dropping
corresponding rows. Relevant columns are selected based on the chosen technique to train the
models effectively.
# Model Training and Evaluation:
For classification tasks, the system splits the dataset into training and testing sets using the
train_test_split function from scikit-learn. The chosen classifier is trained on the training set
and tested on the testing set. The accuracy of the model is calculated using the
accuracy_score function, and additional evaluation metrics such as confusion matrix and
classification report are generated using scikit-learn's functions. The evaluation results are
displayed in a message box for the user to review.
# Analysis and Tips:
The system provides analysis messages specific to each technique, offering insights into the
dataset's characteristics that may impact the model's performance. These analysis messages
highlight factors such as class distribution, feature correlation, outliers, and relevant feature
count. Additionally, the system offers tips to potentially improve the accuracy score by
adjusting certain parameters or exploring alternative techniques. The tips provide guidance on
adjusting parameters like the number of trees for Random Forest, the regularization
parameter for Logistic Regression, the kernel function for SVM, and the value of k for KNN
and Gradient Boosting.
# Visualization:
To aid in result interpretation, the system utilizes data visualization techniques. It generates a
heatmap of the confusion matrix using the seaborn library to provide a visual representation
of the model's performance. For anomaly detection techniques, the system produces scatter
plots to visualize the detected anomalies. The scatter plots depict the relationship between
selected features, with anomalies represented by different colours or markers.
# Graphical User Interface (GUI):
The GUI is created using the tkinter library and provides an intuitive interface for users to
interact with the system. It includes labels for technique selection, entry fields to input
technique numbers, buttons to trigger the classification and anomaly detection processes, and
a browse button to select the dataset. The GUI also displays the selected dataset's path and
provides visualizations and evaluation results in message boxes.
# Required Libraries:
The project utilizes several Python libraries, including tkinter, pandas, matplotlib, seaborn,
scikit-learn, numpy, PIL (Python Imaging Library), and tkinter.filedialog. These libraries
enable various functionalities such as GUI creation, data manipulation, machine learning
algorithm implementation, data visualization, and file handling.
The DDoS Attack Prediction System aims to provide users with a user-friendly interface to
analyse datasets, evaluate models using different techniques, and gain insights into the
dataset's characteristics affecting the chosen techniques' performance. Users can leverage the
system to predict DDoS attacks, classify network traffic, and detect anomalies, enhancing
their ability to manage and secure their network infrastructure.
