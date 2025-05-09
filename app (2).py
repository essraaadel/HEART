
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import random

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

st.title("Top 3 Heart Disease Predictors")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

def remove_outliers_iqr(data, columns, multiplier=1.5):
    df_clean = data.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def build_keras_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Recall'])
    return model

@st.cache_data
def load_and_train_models(file):
    df = pd.read_csv(file)
    df = remove_outliers_iqr(df, df.select_dtypes(include=['int64', 'float64']).columns)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=SEED),
        'Random Forest': RandomForestClassifier(random_state=SEED),
        'SVM': SVC(probability=True, random_state=SEED),
        'Naive Bayes': GaussianNB(),
        'Ridge Classifier': RidgeClassifier(random_state=SEED),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED)
    }

    results = []

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        recall = recall_score(y_test, y_pred)
        results.append((name, recall, pipeline))

    # Neural Network
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    keras_model = build_keras_model(X_train_processed.shape[1])
    keras_model.fit(X_train_processed, y_train, epochs=50, validation_split=0.2, verbose=0)
    y_pred_keras = (keras_model.predict(X_test_processed) > 0.5).astype("int32")
    keras_recall = recall_score(y_test, y_pred_keras)
    results.append(("Neural Network", keras_recall, (preprocessor, keras_model)))

    results.sort(key=lambda x: x[1], reverse=True)
    top3 = results[:3]

    return top3, numeric_features, categorical_features, X

if uploaded_file:
    top3, numeric_features, categorical_features, X_example = load_and_train_models(uploaded_file)

    st.subheader("Top 3 Models by Recall Score:")
    for name, recall, _ in top3:
        st.write(f"**{name}**: Recall = {recall:.4f}")

    st.subheader("Make a Prediction")
    input_data = {}
    for col in numeric_features:
        input_data[col] = st.number_input(f"{col}", value=float(X_example[col].mean()))
    for col in categorical_features:
        input_data[col] = st.selectbox(f"{col}", options=X_example[col].dropna().unique())

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        st.write("Predictions:")
        for name, _, model in top3:
            if name == "Neural Network":
                preprocessor, nn_model = model
                input_transformed = preprocessor.transform(input_df)
                pred = (nn_model.predict(input_transformed) > 0.5).astype("int32")[0][0]
            else:
                pred = model.predict(input_df)[0]
            st.write(f"{name}: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
