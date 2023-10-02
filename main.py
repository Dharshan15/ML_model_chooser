import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

st.title('ML Algoexpert')

st.write("""
# Explore different classifiers on uploaded CSV file
Which one is the best?
""")

uploaded_file = st.file_uploader("Upload a CSV file")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        target_column = st.sidebar.selectbox(
            'Select the target column',
            df.columns.tolist()
        )

        df_X = df.select_dtypes(include=['number'])  # Select numeric columns

        if df_X.empty:
            st.sidebar.write("No numeric columns found in the dataset.")
            st.stop()

        df_y = df[target_column]

        # Encode categorical target column to numeric labels
        label_encoder = LabelEncoder()
        df_y_encoded = label_encoder.fit_transform(df_y)

        st.write("## Dataset")
        st.write('Shape of dataset:', df_X.shape)
        st.write('Number of classes:', len(np.unique(df_y_encoded)))

        classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Random Forest', 'Logistic Regression', 'Naive Bayes', 'Decision Tree')
        )

        def add_parameter_ui(clf_name):
            params = dict()
            if clf_name == 'SVM':
                C = st.sidebar.slider('C', 0.01, 10.0, step=0.01)
                kernel = st.sidebar.selectbox('Kernel', ('linear', 'rbf', 'poly', 'sigmoid'))
                params['C'] = C
                params['kernel'] = kernel
            elif clf_name == 'KNN':
                K = st.sidebar.slider('K', 1, 15)
                params['K'] = K
            elif clf_name == 'Random Forest':
                max_depth = st.sidebar.slider('max_depth', 2, 15)
                n_estimators = st.sidebar.slider('n_estimators', 1, 100)
                params['max_depth'] = max_depth
                params['n_estimators'] = n_estimators
            elif clf_name == 'Logistic Regression':
                C = st.sidebar.slider('C', 0.01, 10.0, step=0.01)
                params['C'] = C
            return params

        params = add_parameter_ui(classifier_name)

        def get_classifier(clf_name, params):
            clf = None
            if clf_name == 'SVM':
                clf = SVC(C=params['C'], kernel=params['kernel'])
            elif clf_name == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=params['K'])
            elif clf_name == 'Random Forest':
                clf = RandomForestClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    random_state=1234
                )
            elif clf_name == 'Logistic Regression':
                clf = LogisticRegression(C=params['C'], solver='liblinear')
            elif clf_name == 'Naive Bayes':
                clf = GaussianNB()
            elif clf_name == 'Decision Tree':
                clf = DecisionTreeClassifier()
            return clf

        clf = get_classifier(classifier_name, params)

        # Preprocess data to handle missing values
        imputer = SimpleImputer(strategy='mean')
        df_X_imputed = pd.DataFrame(imputer.fit_transform(df_X), columns=df_X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            df_X_imputed, df_y_encoded, test_size=0.2, random_state=1234
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.write(f'Classifier = {classifier_name}')
        st.write(f'Accuracy =', acc)

        # Reduce dimensionality to 2 for visualization
        pca = PCA(n_components=2)
        X_projected = pca.fit_transform(df_X_imputed)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]

        st.write("## Dataset Visualization")
        st.write(f"Target column: {target_column}")

        fig = plt.figure()
        plt.scatter(x1, x2, c=df_y_encoded, alpha=0.8, cmap='viridis')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar()

        st.pyplot(fig)

    except pd.errors.EmptyDataError:
        st.sidebar.write("The uploaded file is empty.")