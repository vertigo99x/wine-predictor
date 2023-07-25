import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# Load the wine dataset
wine_data_url = rf'./dataset.csv'

column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines', 'Proline']
data = pd.read_csv(wine_data_url, names=column_names)

# Split the data into features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier without feature selection
clf_without_fs = RandomForestClassifier()
start_time = time.time()
clf_without_fs.fit(X_train, y_train)
end_time = time.time()
time_without_fs = end_time - start_time

# Make predictions
y_pred_without_fs = clf_without_fs.predict(X_test)

# Evaluate the model without feature selection
accuracy_without_fs = accuracy_score(y_test, y_pred_without_fs)

# Apply RFE for feature selection
from sklearn.feature_selection import RFE
rfe = RFE(clf_without_fs, n_features_to_select=5)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Train a random forest classifier with feature selection
clf_with_fs = RandomForestClassifier()
start_time = time.time()
clf_with_fs.fit(X_train_rfe, y_train)
end_time = time.time()
time_with_fs = end_time - start_time

# Make predictions with feature selection
y_pred_with_fs = clf_with_fs.predict(X_test_rfe)

# Evaluate the model with feature selection
accuracy_with_fs = accuracy_score(y_test, y_pred_with_fs)



import matplotlib.pyplot as plt

def main():
    st.title("Wine Classification - Performance Comparison")
    
    # Display accuracy comparison
    plt.figure(figsize=(8, 6))
    metrics = ['Accuracy']
    without_fs_values = [accuracy_without_fs]
    with_fs_values = [accuracy_with_fs]
    labels = ['Without Feature Selection', 'With Feature Selection']
    
    x = range(len(metrics))
    width = 0.2
    plt.bar(x, without_fs_values, width=width, align='center', label='Without Feature Selection')
    plt.bar([i + width for i in x], with_fs_values, width=width, align='center', label='With Feature Selection')
    
    plt.xticks([i + width/2 for i in x], metrics)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.legend(loc='lower right')
    st.pyplot(plt)
    
    # Display time comparison
    plt.figure(figsize=(8, 6))
    metrics = ['Time Taken']
    without_fs_values = [time_without_fs]
    with_fs_values = [time_with_fs]
    labels = ['Without Feature Selection', 'With Feature Selection']
    
    x = range(len(metrics))
    width = 0.2
    plt.bar(x, without_fs_values, width=width, align='center', label='Without Feature Selection')
    plt.bar([i + width for i in x], with_fs_values, width=width, align='center', label='With Feature Selection')
    
    plt.xticks([i + width/2 for i in x], metrics)
    plt.ylabel('Time Taken (s)')
    plt.title('Model Training Time Comparison')
    plt.legend(loc='upper right')
    st.pyplot(plt)
    
    st.write("Random Forest Classifier Accuracy (Without Feature Selection):", accuracy_without_fs)
    st.write("Random Forest Classifier Accuracy (With Feature Selection):", accuracy_with_fs)
    st.write("Time Taken (Without Feature Selection):", time_without_fs, "s")
    st.write("Time Taken (With Feature Selection):", time_with_fs, "s")

if __name__ == '__main__':
    main()

