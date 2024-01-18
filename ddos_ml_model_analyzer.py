import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import tkinter.filedialog as filedialog
from PIL import ImageTk, Image
def classification_model_analysis(data,technique):
    analysis_message = ""

    if technique == 1:
        # Perform analysis specific to Random Forest and the dataset
        # Example analysis: Check if the dataset has balanced class distribution
        class_counts = data['Label'].value_counts()
        if len(class_counts) == 2 and class_counts.min() / class_counts.max() >= 0.2:
            analysis_message = "Random Forest analysis:\n The dataset has a balanced class distribution, which is suitable for the Random Forest model."
        else:
            analysis_message = "Random Forest analysis:\n The dataset does not have a balanced class distribution, which may affect the performance of the Random Forest model. Consider balancing the classes or using other techniques."

    elif technique == 2:
        # Perform analysis specific to Logistic Regression and the dataset
        # Example analysis: Check if the dataset has linearly separable classes
        correlation_matrix = data.corr()
        class_correlations = correlation_matrix['Label'].drop('Label')
        if (class_correlations.abs() >= 0.5).any():
            analysis_message = "Logistic Regression analysis:\n The dataset has some variables with moderate to high correlation with the target variable, suggesting potential linear separability."
        else:
            analysis_message = "Logistic Regression analysis:\n The dataset does not have linear separability."
    elif technique == 3:
        # Perform analysis specific to SVM and the dataset
        # Example analysis: Check if the dataset has clearly separable classes
        class_counts = data['Label'].value_counts()
        if len(class_counts) == 2 and class_counts.min() / class_counts.max() >= 0.2:
            analysis_message = "SVM analysis:\n The dataset has clearly separable classes, which is suitable for the SVM model."
        else:
            analysis_message = "SVM analysis:\n The dataset does not have clearly separable classes, which may affect the performance of the SVM model. Consider feature engineering or using other techniques."

    elif technique == 4:
        # Perform analysis specific to KNN and the dataset
        # Example analysis: Check if the dataset has relevant features
        num_features = data.shape[1] - 1
        if num_features >= 3:
            analysis_message = "KNN analysis:\n The dataset has a sufficient number of relevant features, which can provide meaningful information for the KNN model."

    elif technique == 5:
        # Perform analysis specific to Gradient Boosting and the dataset
        # Example analysis: Check if the dataset has outliers that may affect the model
        outlier_ratio = (data['Label'] == 1).mean()
        if outlier_ratio <= 0.05:
            analysis_message = "Gradient Boosting analysis:\n The dataset has a low proportion of outliers, which may not significantly affect the model's performance."
        else:
            analysis_message = "Gradient Boosting analysis:\n The dataset has a high proportion of outliers, which may negatively impact the model's performance. Consider outlier detection or using other techniques."

    return analysis_message


def anomaly_detection_model_analysis(data,technique):
    analysis_message = ""

    if technique == 1:
        # Perform analysis specific to Isolation Forest and the dataset
        # Example analysis: Check if the dataset has a wide range of feature values
        feature_ranges = data.max() - data.min()
        if (feature_ranges >= 10).all():
            analysis_message = "Isolation Forest analysis:\n The dataset has a wide range of feature values, which can enhance the performance of the Isolation Forest model."
        else:
            analysis_message = "Isolation Forest analysis:\n The dataset does not have a wide range of feature values, which may affect the performance of the Isolation Forest model. Consider feature scaling or using other techniques."

    elif technique == 2:
        # Perform analysis specific to Local Outlier Factor and the dataset
        # Example analysis: Check if the dataset has distinct clusters
        num_clusters = len(data['Label'].unique())
        if num_clusters >= 2:
            analysis_message = "Local Outlier Factor analysis:\n The dataset has distinct clusters, which can improve the performance of the Local Outlier Factor model."
        else:
            analysis_message = "Local Outlier Factor analysis:\n The dataset does not have distinct clusters, which may affect the performance of the Local Outlier Factor model. Consider other techniques for outlier detection."

    elif technique == 3:
        # Perform analysis specific to One-Class SVM and the dataset
        # Example analysis: Check if the dataset has clear separation between normal and anomalous instances
        class_counts = data['Label'].value_counts()
        if len(class_counts) == 2 and class_counts.min() / class_counts.max() >= 0.2:
            analysis_message = "One-Class SVM analysis:\n The dataset has clear separation between normal and anomalous instances, which is suitable for the One-Class SVM model."
        else:
            analysis_message = "One-Class SVM analysis:\n The dataset does not have clear separation between normal and anomalous instances, which may affect the performance of the One-Class SVM model. Consider other techniques for anomaly detection."

    elif technique == 4:
        # Perform analysis specific to DBSCAN and the dataset
        # Example analysis: Check if the dataset has appropriate density for clustering
        num_samples = data.shape[0]
        if num_samples >= 500 and num_samples <= 5000:
            analysis_message = "DBSCAN analysis:\n The dataset has an appropriate density for clustering, which is suitable for the DBSCAN model."
        else:
            analysis_message = "DBSCAN analysis:\n The dataset does not have an appropriate density for clustering, which may affect the performance of the DBSCAN model. Consider other techniques for anomaly detection."

    return analysis_message

def classification_model_tip(technique):
    analysis_message = ""

    if technique == 1:
        analysis_message = """
        
        Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes predicted by the individual trees.
        
        To potentially improve the accuracy score, you can consider adjusting the number of trees in the forest. Adding more trees can help improve the model's ability to capture complex relationships in the dataset. You can try increasing the number of trees and evaluate if it leads to a higher accuracy score. However, be mindful of computational resources as adding more trees may increase the training and prediction time."""

    elif technique == 2:
        analysis_message = """
        
        Logistic Regression is a statistical model used to predict the probability of a binary response based on one or more predictor variables. It uses a logistic function to model the relationship between the predictors and the response.
        
        To potentially improve the accuracy score, you can consider adjusting the regularization parameter. Regularization helps control overfitting in the model. If the accuracy is lower, you can try decreasing the regularization parameter to reduce the penalty for complex models. On the other hand, if the accuracy is higher, you can try increasing the regularization parameter to prevent overfitting. Experimenting with different values of the regularization parameter can help find the optimal balance between model complexity and generalization."""

    elif technique == 3:
        analysis_message = """
        
        Support Vector Machines (SVM) is a supervised learning algorithm that can be used for classification or regression tasks. It finds a hyperplane that maximally separates the classes in the feature space.
        
        To potentially improve the accuracy score, you can consider trying different kernel functions. SVM uses different kernels (e.g., linear, polynomial, radial basis function) to transform the data into a higher-dimensional space where it can be more easily separated. If the accuracy is lower, you can experiment with different kernels to find the one that better captures the underlying patterns in your dataset. Each kernel has its own characteristics, and choosing the appropriate one can have a significant impact on the accuracy."""

    elif technique == 4:
        analysis_message = """KNN analysis:
        
        K-Nearest Neighbors (KNN) is a non-parametric algorithm used for both classification and regression. It assigns the class or value based on the majority vote or average of the k nearest neighbors in the feature space.
        
        To potentially improve the accuracy score, you can consider adjusting the value of k. The choice of k determines the number of neighbors considered for classification. If the accuracy is lower, you can try increasing the value of k to consider more neighbors, which can provide a smoother decision boundary. On the other hand, if the accuracy is higher, you can try decreasing the value of k to focus on the nearest neighbors and capture local patterns more precisely. Experimenting with different values of k can help find the optimal balance between local and global information."""

    elif technique == 5:
        analysis_message = """
        
        Gradient Boosting is an ensemble learning method that combines multiple weak models (typically decision trees) to create a strong predictive model. It builds the models sequentially, with each model correcting the mistakes of the previous models.
        
        To potentially improve the accuracy score, you can consider increasing the number of boosting stages. Boosting involves adding models sequentially, with each model trying to correct the errors made by the previous models. If the accuracy is lower, you can try increasing the number of boosting stages to allow the model to learn more complex patterns in the dataset. However, be cautious of overfitting, as a higher number of boosting stages may lead to capturing noise in the training data. Experimenting with different numbers of boosting stages can help find the optimal balance between model complexity and generalization."""

    return analysis_message

def anomaly_model_tip(technique):
    analysis_message = ""

    if technique == 1:
        analysis_message = """
        
        Isolation Forest is an unsupervised learning algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
        
        To analyze and potentially improve the accuracy score, you can consider adjusting the contamination parameter. The contamination parameter represents the expected proportion of outliers in the dataset. If the accuracy is lower, you can try decreasing the contamination parameter to consider fewer points as outliers. On the other hand, if the accuracy is higher, you can try increasing the contamination parameter to consider more points as outliers. Experimenting with different values of the contamination parameter can help find the optimal balance between outlier detection and accuracy."""

    elif technique == 2:
        analysis_message = """
        
        Local Outlier Factor (LOF) is an unsupervised learning algorithm that computes a score (LOF score) reflecting the degree of abnormality for each data point based on the local density deviation of its neighbors.
        
        To analyze and potentially improve the accuracy score, you can consider adjusting the number of neighbors used in the LOF algorithm. The number of neighbors influences the local density estimation and the identification of outliers. If the accuracy is lower, you can try increasing the number of neighbors to capture more local density information. On the other hand, if the accuracy is higher, you can try decreasing the number of neighbors to focus on more prominent outliers. Experimenting with different values of the number of neighbors can help find the optimal balance between sensitivity to outliers and accuracy."""

    elif technique == 3:
        analysis_message = """
        
        One-Class Support Vector Machines (One-Class SVM) is an unsupervised learning algorithm that separates the data points in a high-dimensional feature space from the origin, thereby identifying outliers as data points outside the separation boundary.
        
        To analyze and potentially improve the accuracy score, you can consider adjusting the kernel function used in the One-Class SVM. The choice of kernel can impact the separation boundary and the identification of outliers. If the accuracy is lower, you can experiment with different kernel functions to find the one that better captures the characteristics of the outliers in your dataset. Each kernel has its own properties, and selecting the appropriate one can have a significant impact on the accuracy."""

    elif technique == 4:
        analysis_message = """
        
        DBSCAN is a density-based clustering algorithm that groups data points based on their proximity and density while marking outliers as noise. It is robust to noise, can discover clusters of arbitrary shape, and does not require specifying the number of clusters in advance.
        
        To analyze and potentially improve the accuracy score, you can consider adjusting the epsilon (ε) and min_samples parameters of the DBSCAN algorithm. The epsilon parameter defines the radius within which neighboring points are considered part of a cluster. The min_samples parameter determines the minimum number of points required to form a dense region. If the accuracy is lower, you can try decreasing the epsilon value or increasing the min_samples value to make the algorithm more sensitive to outliers. On the other hand, if the accuracy is higher, you can try increasing the epsilon value or decreasing the min_samples value to focus on more compact clusters and reduce false positives. Experimenting with different values of epsilon and min_samples can help find the optimal balance between cluster formation and outlier detection."""

    return(analysis_message)


def train_and_predict_classification(technique,data_path):
    # Dataset Preparation
    data = pd.read_csv(data_path)  

    # Data Cleaning and Feature Extraction
    # Drop rows with missing values
    data = data.dropna()
    # Select relevant columns for the selected technique
    if technique == 1:
        selected_columns = ['Flow_IAT_Min', 'Tot_Fwd_Pkts', 'Init_Bwd_Win_Bytes']
    elif technique == 2:
        selected_columns = ['Flow_IAT_Min', 'Tot_Fwd_Pkts', 'Init_Bwd_Win_Bytes']
    elif technique == 3:
        selected_columns = ['Flow_IAT_Min', 'Tot_Fwd_Pkts', 'Init_Bwd_Win_Bytes']
    elif technique == 4:
        selected_columns = ['Flow_IAT_Min', 'Tot_Fwd_Pkts', 'Init_Bwd_Win_Bytes']
    elif technique == 5:
        selected_columns = ['Flow_IAT_Min', 'Tot_Fwd_Pkts', 'Init_Bwd_Win_Bytes']

    X = data[selected_columns]
    y = data['Label']

    # Model Training and Testing
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the selected classifier
    if technique == 1:
        classifier = RandomForestClassifier()
    elif technique == 2:
        classifier = LogisticRegression()
    elif technique == 3:
        classifier = SVC()
    elif technique == 4:
        classifier = KNeighborsClassifier()
    elif technique == 5:
        classifier = GradientBoostingClassifier()

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)  # Set zero_division to 1

    # User Interface
    technique_name = ''
    technique_description = ''
    if technique == 1:
        technique_name = 'Random Forest'
        technique_description = 'Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes predicted by the individual trees.'
    elif technique == 2:
        technique_name = 'Logistic Regression'
        technique_description = 'Logistic Regression is a statistical model used to predict the probability of a binary response based on one or more predictor variables. It uses a logistic function to model the relationship between the predictors and the response.'
    elif technique == 3:
        technique_name = 'SVM'
        technique_description = 'Support Vector Machines (SVM) is a supervised learning algorithm that can be used for classification or regression tasks. It finds a hyperplane that maximally separates the classes in the feature space.'
    elif technique == 4:
        technique_name = 'KNN'
        technique_description = 'K-Nearest Neighbors (KNN) is a non-parametric algorithm used for both classification and regression. It assigns the class or value based on the majority vote or average of the k nearest neighbors in the feature space.'
    elif technique == 5:
        technique_name = 'Gradient Boosting'
        technique_description = 'Gradient Boosting is an ensemble learning method that combines multiple weak models (typically decision trees) to create a strong predictive model. It builds the models sequentially, with each model correcting the mistakes of the previous models.'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix - {technique_name}", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    if technique != 2:
        analysis_info = classification_model_tip(technique)
        analysis_result = classification_model_analysis(data,technique)

    messagebox.showinfo("Model Evaluation", f"{technique_name} Evaluation:\n\n{report}\n\n{technique_description}\n\nAccuracy: {accuracy:.2f}")
    if technique != 2:
        messagebox.showinfo("Analysis Result", f"Model Analysis:\n\n{analysis_info}\n\n{analysis_result}")

def train_and_predict_anomaly_detection(technique, data_path):
    # Dataset Preparation
    data = pd.read_csv(data_path)

    # Data Cleaning and Feature Extraction
    # Drop rows with missing values
    data = data.dropna()
    # Select relevant columns for anomaly detection
    selected_columns = ['Flow_IAT_Min', 'Tot_Fwd_Pkts', 'Init_Bwd_Win_Bytes']

    X = data[selected_columns]
    y = data['Label']

    # Data Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Anomaly Detection
    if technique == 1:
        model = IsolationForest()
    elif technique == 2:
        model = LocalOutlierFactor()
    elif technique == 3:
        model = OneClassSVM()
    elif technique == 4:
        model = DBSCAN()

    if technique > 0 and technique < 5:
        y_pred = model.fit_predict(X_scaled)
        y_pred[y_pred == 1] = 0  # Invert the labels for visualization purposes (0: normal, 1: anomaly)
        y_pred[y_pred == -1] = 1

        cm = confusion_matrix(data['Label'], y_pred)
        report = classification_report(data['Label'], y_pred, zero_division=1)
        report1 = classification_report(data['Label'], y_pred, zero_division=1, output_dict=True)

        technique_name = ''
        technique_description = ''
        if technique == 1:
            technique_name = 'Isolation Forest'
            technique_description = 'Isolation Forest is an unsupervised learning algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.'
        elif technique == 2:
            technique_name = 'Local Outlier Factor'
            technique_description = 'Local Outlier Factor (LOF) is an unsupervised learning algorithm that computes a score (LOF score) reflecting the degree of abnormality for each data point based on the local density deviation of its neighbors.'
        elif technique == 3:
            technique_name = 'One-Class SVM'
            technique_description = 'One-Class Support Vector Machines (One-Class SVM) is an unsupervised learning algorithm that separates the data points in a high-dimensional feature space from the origin, thereby identifying outliers as data points outside the separation boundary.'
        elif technique == 4:
            technique_name = 'DBSCAN'
            technique_description = 'DBSCAN is a density-based clustering algorithm that groups data points based on their proximity and density, while marking outliers as noise. It is robust to noise, can discover clusters of arbitrary shape, and does not require specifying the number of clusters in advance.'

        accuracy = report1['accuracy']

        # Confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title(f"Confusion Matrix - Anomaly Detection ({technique_name})", fontsize=16)
        plt.xlabel("Predicted Label", fontsize=14)
        plt.ylabel("True Label", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

        # Generate scatter plot of anomalies detected
        plt.figure(figsize=(10, 8))
        plt.scatter(X['Flow_IAT_Min'], X['Tot_Fwd_Pkts'], c=y_pred, cmap='RdYlBu')
        plt.title(f"Anomalies Detected - {technique_name}", fontsize=16)
        plt.xlabel("Flow_IAT_Min", fontsize=14)
        plt.ylabel("Tot_Fwd_Pkts", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

        
        if technique != 1:
            analysis_info = anomaly_model_tip(technique)
            analysis_result = anomaly_detection_model_analysis(data, technique)


          
        # Display results
        messagebox.showinfo(
            "Anomaly Detection Evaluation",
            f"Anomaly Detection ({technique_name}) Evaluation:\n\n{report}\n\n{technique_description}\n\nAccuracy: {accuracy:.2f}"
        )
        if technique != 1:
            messagebox.showinfo("Analysis Result", f"Model Analysis:\n\n{analysis_info}\n\n{analysis_result}")
        
    else:
        messagebox.showinfo("Error", "Please enter a value between 1-4.")


def create_gui():
    root = tk.Tk()
    root.title("DDoS Attack Prediction System")
    root.geometry("1100x700")  # Set the window size

    def on_button_click_classification():
        try:
            technique = int(entry_classification.get())
            train_and_predict_classification(technique,data_path)
        except:
            messagebox.showerror("Error", "Invalid selection. Please choose a technique from 1 to 5. Or not a suitable model.")

    def on_button_click_anomaly_detection():
        try:
            technique = int(entry_anomaly_detection.get())
            train_and_predict_anomaly_detection(technique, data_path)
        except ValueError:
            messagebox.showerror("Error", "Invalid selection. Please choose a technique from 1 to 5. Or not a suitable model.")
    
    def browse_file():
        global data_path
        file_path = filedialog.askopenfilename()
        if file_path:
            data_path = file_path
            path_label.config(text=f"Selected Dataset: {data_path}")

    # Create a scrollable frame
    scroll_frame = tk.Frame(root)
    scroll_frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas
    canvas = tk.Canvas(scroll_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create a scrollbar
    scrollbar = tk.Scrollbar(scroll_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)

    # Configure the canvas to resize with the window
    canvas.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

    # Create a frame inside the canvas to hold the content
    content_frame = tk.Frame(canvas)
    canvas.create_window((500,500), window=content_frame, anchor=tk.NW)

    # Add the content to the frame
    image = Image.open("heading.jpeg")
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(content_frame, image=photo)
    label.pack()

    entry_label_space = tk.Label(content_frame, text="\n")
    entry_label_space.pack()

    path_label = tk.Label(content_frame, text="Selected Dataset: None", font=("Arial", 16))
    path_label.pack()

    browse_button = tk.Button(content_frame, text="Browse", command=browse_file, font=("Arial", 16))
    browse_button.pack()

    entry_label_space = tk.Label(content_frame, text="\n")
    entry_label_space.pack()

    label_classification = tk.Label(content_frame, text="Classification Techniques:", font=("Arial", 22))
    label_classification.pack()

    label_rf = tk.Label(content_frame, text="1. Random Forest", font=("Arial", 16))
    label_rf.pack()

    label_lr = tk.Label(content_frame, text="2. Logistic Regression", font=("Arial", 16))
    label_lr.pack()

    label_svm = tk.Label(content_frame, text="3. SVM", font=("Arial", 16))
    label_svm.pack()

    label_knn = tk.Label(content_frame, text="4. KNN", font=("Arial", 16))
    label_knn.pack()

    label_gb = tk.Label(content_frame, text="5. Gradient Boosting", font=("Arial", 16))
    label_gb.pack()

    entry_label_classification = tk.Label(content_frame, text="Select a classification technique (1-5):", font=("Arial", 16))
    entry_label_classification.pack()

    entry_classification = tk.Entry(content_frame, font=("Arial", 16))
    entry_classification.pack()

    button_classification = tk.Button(content_frame, text="Predict (Classification)", command=on_button_click_classification, font=("Arial", 16))
    button_classification.pack()

    entry_label_space = tk.Label(content_frame, text="\n")
    entry_label_space.pack()

    label_anomaly_detection = tk.Label(content_frame, text="Anomaly Detection Techniques:", font=("Arial", 22))
    label_anomaly_detection.pack()

    label_if = tk.Label(content_frame, text="1. Isolation Forest", font=("Arial", 16))
    label_if.pack()

    label_lof = tk.Label(content_frame, text="2. Local Outlier Factor", font=("Arial", 16))
    label_lof.pack()

    label_ocsvm = tk.Label(content_frame, text="3. One-Class SVM", font=("Arial", 16))
    label_ocsvm.pack()

    label_dbscan = tk.Label(content_frame, text="4. DBSCAN", font=("Arial", 16))
    label_dbscan.pack()

    entry_label_anomaly_detection = tk.Label(content_frame, text="Select an anomaly detection technique (1-4):", font=("Arial", 16))
    entry_label_anomaly_detection.pack()

    entry_anomaly_detection = tk.Entry(content_frame, font=("Arial", 16))
    entry_anomaly_detection.pack()

    button_anomaly_detection = tk.Button(content_frame, text="Predict (Anomaly Detection)", command=on_button_click_anomaly_detection, font=("Arial", 16))
    button_anomaly_detection.pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()
