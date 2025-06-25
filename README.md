# **Credit Card Fraud Detection Project**

## **Overview**

This project focuses on building and evaluating machine learning models to detect fraudulent credit card transactions from a highly imbalanced dataset. The primary objective is to minimize the **False Negative Rate (FNR)**, as missing actual fraudulent transactions is typically more costly than flagging legitimate transactions as suspicious (false positives).  
The project explores and enhances three main types of models:

1. **Logistic Regression**  
2. **K-means Clustering**  
3. **Fully Connected Neural Network**

The dataset used is the anonymized [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## **Project Structure**

ml-fraud-detection-project/  
├── data/  
│   └── creditcard.csv       \# The raw dataset (download separately)  
├── src/  
│   ├── \_\_init\_\_.py            \# Marks 'src' as a Python package  
│   ├── data\_preprocessing.py  \# Handles data loading, exploration, splitting, scaling, SMOTE, PCA  
│   ├── logistic\_regression\_model.py \# Implements Logistic Regression models (vanilla, SMOTE, balanced, tuned pipeline)  
│   ├── k\_means\_model.py       \# Implements K-means Clustering model  
│   ├── neural\_network\_model.py\# Implements Neural Network model (fixed, advanced with custom loss)  
│   └── utils.py               \# Contains helper functions for evaluation and plotting  
├── notebooks/                 \# Optional: For Jupyter Notebooks (e.g., run\_all\_models.ipynb if used)  
├── plots/                     \# Automatically created to save generated model evaluation plots  
├── run\_phase1.py              \# Script to verify Phase 1: Setup and Data Loading  
├── run\_phase2.py              \# Script to run Phase 2: Logistic Regression Models (Vanilla, SMOTE, Balanced)  
├── run\_phase3.py              \# Script to run Phase 3: K-means Clustering Model, and later overwritten for advanced LR/NN phases  
├── run\_phase4.py              \# Script to run Phase 4: Fixed Neural Network Model, and later overwritten for advanced NN phases  
└── requirements.txt           \# List of Python dependencies

## **Setup Instructions**

1. **Clone the Repository:**  
   git clone https://github.com/georgymh/ml-fraud-detection.git \# (Assuming this is your repo URL)  
   cd ml-fraud-detection-project

   *(Note: If you downloaded as a ZIP, adjust the folder name accordingly.)*  
2. Download the Dataset:  
   Download creditcard.csv from Kaggle and place it inside the data/ directory.  
   * [Kaggle Dataset: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
3. **Create and Activate a Python Virtual Environment (Recommended):**  
   python3 \-m venv fraud\_env  
   \# On macOS/Linux:  
   source fraud\_env/bin/activate  
   \# On Windows (Command Prompt):  
   .\\fraud\_env\\Scripts\\activate.bat  
   \# On Windows (PowerShell):  
   .\\fraud\_env\\Scripts\\Activate.ps1

   Your terminal prompt should show (fraud\_env) when active.  
4. **Install Dependencies:**  
   pip install \-r requirements.txt

## **How to Run the Project**

Due to file overwriting during the development process, the phases are now mapped to the following scripts:

* **To run Phase 1 (Setup and Data Loading):**  
  python run\_phase1.py

* **To run Phase 2 (Logistic Regression \- Vanilla, SMOTE, Balanced):**  
  python run\_phase2.py

* **To run Phase 3 (K-means Clustering Model):**  
  python run\_phase3.py

  *(Note: If you overwrote run\_phase3.py with code from later phases, you would need to revert it to its original Phase 3 content to run this specific phase.)*  
* **To run Phase 4 (Fixed Neural Network Model):**  
  python run\_phase4.py

  *(Note: If you overwrote run\_phase4.py with code from later phases, you would need to revert it to its original Phase 4 content to run this specific phase.)*  
* To run the Advanced Logistic Regression (Phase 5/8) and Advanced Neural Network (Phase 6/7):  
  The code for these advanced models has been placed in run\_phase3.py and run\_phase4.py respectively in your setup.  
  * To run the **Advanced Logistic Regression (Phase 5/8 \- SMOTE in Pipeline)**:  
    python run\_phase3.py

  * To run the **Advanced Neural Network (Phase 6/7 \- Custom Weighted Loss)**:  
    python run\_phase4.py

The scripts will print outputs to the terminal, and all generated plots (confusion matrices, accuracy/loss curves, feature distributions) will be saved as .png files in the plots/ directory within your project root.

## **Model Implementations & Key Results**

### **Data Characteristics**

The dataset contains 284,807 transactions with 30 features (V1-V28 are PCA-transformed, Time, Amount). It is highly imbalanced, with only **0.173%** (492 transactions) being fraudulent.

### **Phase 2: Logistic Regression Models (Vanilla, SMOTE, Balanced Weights)**

* **Vanilla LR:** High Accuracy (\~99.9%), but very high FNR (missed \~50% of frauds), making it unsuitable.  
* **LR with SMOTE & Scaling:** Improved FNR significantly to \~0.1163.  
* **LR with Balanced Class Weights:** Achieved a competitive FNR of \~0.1221, balancing positive and negative class importance during training.

### **Phase 3: K-means Clustering Model**

* Performed poorly (Accuracy \~54%), with a high FNR for fraud detection due to its unsupervised nature and reliance on feature similarities. Not suitable for this problem as a direct classifier.

### **Phase 4: Fixed Architecture Neural Network**

* Initial attempt showed unstable training (high and fluctuating loss) and a high FNR of \~0.3765, struggling with the imbalance even after SMOTE.

### **Phase 5 & 8: Tuned Logistic Regression Pipeline (Best Approach for LR)**

* **Methodology:** Integrated StandardScaler and SMOTE into a Pipeline for robust cross-validation (GridSearchCV tuned C for optimal Recall).  
* **Best Parameters:** {'logisticregression\_\_C': 0.01, 'logisticregression\_\_solver': 'liblinear'}.  
* **Test Set Performance:**  
  * Accuracy: \~0.9787  
  * Recall (Fraud): \~0.8721  
  * Precision (Fraud): \~0.0667  
  * **FNR: \~0.1279** (12.79% of actual frauds missed)  
  * F1-Score: \~0.1239  
* **Analysis:** This robust pipeline confirmed strong fraud detection capabilities (high recall/low FNR) for Logistic Regression, but still with a relatively high number of false positives (low precision).

### **Phase 6 & 7: Advanced Neural Network with Custom Weighted Loss**

* **Methodology:** Implemented a deeper network architecture with Dropout, EarlyStopping, ReduceLROnPlateau, and a **Custom Weighted Binary Cross-Entropy Loss** function to explicitly weigh the minority class more heavily during training.  
* **Test Set Performance:**  
  * Accuracy: \~0.9988  
  * Recall (Fraud): \~0.7901  
  * Precision (Fraud): \~0.6305  
  * **FNR: \~0.2099** (20.99% of actual frauds missed)  
  * F1-Score: \~0.7014  
* **Analysis:** This model showed **dramatic improvement** over the initial NN. While its FNR is slightly higher than the tuned LR, it achieved a **significantly higher Precision** and **much higher F1-Score**. This indicates a better balance between catching frauds and minimizing false alarms. In many real-world scenarios, this trade-off is highly desirable to reduce operational overhead from too many false positives.

## **Conclusion**

Both the **Tuned Logistic Regression Pipeline** and the **Advanced Neural Network with Custom Weighted Loss** are strong contenders for credit card fraud detection on this dataset.

* If **missing *absolutely no* fraud is the highest priority**, even at the cost of many false alarms, further threshold tuning on the Logistic Regression model could yield perfect recall.  
* If a **better balance between catching fraud and minimizing false alarms (reducing manual review burden)** is desired, the **Advanced Neural Network** is the superior choice due to its significantly higher precision and F1-score.

The choice between these models depends on the specific business requirements and the associated costs of false negatives vs. false positives.

## **Future Improvements**

* **Explore other Advanced Models:** Implement and evaluate tree-based ensemble methods like Random Forest or Gradient Boosting Machines (XGBoost, LightGBM), which often perform exceptionally well on tabular, imbalanced data.  
* **More Extensive Hyperparameter Tuning:** Utilize more advanced tuning strategies (e.g., RandomizedSearchCV for NNs) over a wider range of parameters.  
* **Anomaly Detection Techniques:** Investigate methods like One-Class SVM or Autoencoders if the problem is framed more as anomaly detection.  
* **Advanced Sampling Strategies:** Experiment with other imblearn oversampling/undersampling techniques (ADASYN, BorderlineSMOTE, NearMiss).  
* **Cost-Sensitive Learning:** Directly incorporate misclassification costs into the model's loss function or evaluation.
