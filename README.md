**SMS Spam Detection**

This project aims to classify SMS messages as spam or ham (not spam) using a dataset of SMS messages. The model uses a Naive Bayes classifier to predict whether a message is spam. The project includes data preprocessing, feature engineering, model training, evaluation, and visualization.

**Dataset**

The dataset used in this project is the spam.csv file, 
which contains SMS messages labeled as spam or ham. 

**The Dataset is colleced from Kaggle:-**

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

**The columns are:**

v1: Label (ham or spam)

v2: SMS message

**Project Structure**

data_loading: Load the dataset and handle missing values.

data_preprocessing: Clean text data, tokenize, remove stopwords, and apply stemming.

data_visualization: Visualize class distribution and message lengths.

feature_engineering: Create new features and vectorize text using TF-IDF.

model_training: Split the data, apply oversampling, initialize and train a Naive Bayes classifier.

model_evaluation: Evaluate the model using accuracy, precision, recall, F1 score, and other metrics.



**Installation**

To run this project, ensure you have the following libraries installed:

numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
imblearn


**You can install these libraries using pip:**

pip install <'library name'>


**Results**

The model's performance is evaluated based on various metrics, 
providing a comprehensive understanding of its effectiveness in detecting spam messages.


**License**

This project is licensed under the MIT License. See the LICENSE file for more details.
