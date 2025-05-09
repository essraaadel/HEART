#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error,classification_report,confusion_matrix,ConfusionMatrixDisplay


# In[2]:


df=pd.read_csv(r"D:\Anaconda\1-Jupyter\Heart Project\heart.csv")
df


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


plt.figure(figsize=(14, 10))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='Spectral', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[7]:


print("No.Duplicated=",df.duplicated().sum())
df.drop_duplicates(inplace=True)
df


# In[8]:


sns.countplot(data=df, x='HeartDisease')
plt.title("Distribution of Heart Disease")
plt.show()


# In[9]:


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15, 8))
df[numeric_cols].boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot of Numeric Features")
plt.show()


# In[10]:


## الخطوة 4: التحضير للمعالجة
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# التعرف على أنواع الأعمدة
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

print("Numeric Features:", num_features)
print("Categorical Features:", cat_features)


# In[11]:


## الخطوة 5: بناء بايبلاين المعالجة المسبقة
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('power', PowerTransformer())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_features),
    ('cat', categorical_pipeline, cat_features)
])


# In[12]:


## الخطوة 6: تقسيم البيانات للتدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


## الخطوة 7: تعريف النماذج
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Lasso (Logistic L1)': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000),
    'Ridge (Logistic L2)': LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
}


# In[14]:


## الخطوة 8: تدريب النماذج وتقييم الأداء
results = {}

for name, model in models.items():
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("*********************************************************\n")


# In[15]:


## تدريب النماذج وتقييم الأداء + مصفوفة الالتباس
results = {}

for name, model in models.items():
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # عرض مصفوفة الالتباس
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_ if hasattr(clf, 'classes_') else [0,1])
    disp.plot(cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()


# In[16]:


## الخطوة 9: مقارنة أداء النماذج
plt.figure(figsize=(10,6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.ylim(0.7, 1.0)
plt.show()


# In[34]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[36]:


# تجهيز البيانات لنموذج Keras (بعد التحويل المسبق)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# بناء النموذج
keras_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# تجميع النموذج
keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# التدريب مع EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = keras_model.fit(X_train_processed, y_train, epochs=100, batch_size=32,
                          validation_split=0.2, callbacks=[early_stop], verbose=0)

# التقييم
loss, acc = keras_model.evaluate(X_test_processed, y_test, verbose=0)
print(f"Keras Neural Network Accuracy: {acc:.4f}")
results['Keras Neural Network'] = acc


# In[38]:


# تحديث الرسم البياني ليشمل Keras
plt.figure(figsize=(10,6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Comparison (Including Keras)")
plt.ylim(0.7, 1.0)
plt.show()


# In[40]:


# التنبؤ بالقيم الاحتمالية وتحويلها إلى 0 أو 1
y_pred_keras = (keras_model.predict(X_test_processed) > 0.5).astype("int32")

# تقييم الدقة
loss, acc = keras_model.evaluate(X_test_processed, y_test, verbose=0)
print(f"\nKeras Neural Network Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred_keras))

# مصفوفة الالتباس
cm_keras = confusion_matrix(y_test, y_pred_keras)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_keras, display_labels=[0,1])
disp.plot(cmap='Blues')
plt.title("Keras Neural Network - Confusion Matrix")
plt.show()

results['Keras Neural Network'] = acc


# In[ ]:




