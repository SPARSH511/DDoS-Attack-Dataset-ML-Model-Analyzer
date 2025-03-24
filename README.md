# DDoS Attack ML Model Analyzer

**Overview:** <br>
The DDoS ML Model Analyzer is a Python-based tool with a Tkinter GUI that allows users to analyze DDoS attack datasets (.csv) and determine the best classification or anomaly detection model.

---

**1. Classification Techniques:**
- **Random Forest:** Checks class balance; suggests class balancing if needed.
- **Logistic Regression:** Assesses linear separability via correlation analysis.
- **SVM (Support Vector Machines):** Examines class separation; suggests feature engineering if unclear.
- **KNN (K-Nearest Neighbours):** Analyzes feature sufficiency for accurate predictions.
- **Gradient Boosting:** Detects outliers; suggests outlier detection techniques if needed.

**2. Anomaly Detection Techniques:**
- **Isolation Forest:** Evaluates feature value range to enhance performance.
- **Local Outlier Factor (LOF):** Checks for distinct clusters to improve accuracy.
- **One-Class SVM:** Determines separation between normal and anomalous instances.
- **DBSCAN:** Analyzes dataset density for effective clustering.

**3. Dataset Selection & Preprocessing:**
- Users select a dataset (.csv) via a browse button.
- Data is loaded using pandas.
- Missing values are removed.
- Relevant columns are selected based on the chosen model.

**4. Model Training & Evaluation:**
- Dataset is split into training/testing sets.
- Chosen classifier is trained and evaluated.
- Accuracy, confusion matrix, and classification reports are generated.
- Results are displayed in a message box.

**5. Analysis & Improvement Tips:**
- Provides dataset insights (class distribution, correlations, outliers, feature count).
- Suggests tuning parameters (trees for Random Forest, regularization for Logistic Regression, kernel for SVM, k-value for KNN & Gradient Boosting).

**6. Visualization:**
- **Classification:** Heatmap of the confusion matrix (using Seaborn).
- **Anomaly Detection:** Scatter plots for detected anomalies.

**7. Graphical User Interface (GUI):**
- Built using Tkinter.
- Provides options to select dataset, enter model choice, trigger analysis, and display results.

**8. Required Libraries:**
- `tkinter`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `numpy`, `PIL`, `tkinter.filedialog`

**Purpose:**
This tool helps users analyze DDoS datasets, predict attacks, classify network traffic, and detect anomalies, improving network security and management.


