from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler

# Database connection configuration using SQLAlchemy
sqluser = 'mimicuser'
password = '123'
host = 'localhost'
dbname = 'mimic'
port = '5432'
schema_name = 'mimiciii'
database_url = f"postgresql://{sqluser}:{password}@{host}:{port}/{dbname}"
engine = create_engine(database_url)

# Step 1: Data Extraction
query = f"""
SELECT pres.subject_id, pres.hadm_id, pres.drug, adm.hospital_expire_flag
FROM {schema_name}.prescriptions pres
JOIN {schema_name}.admissions adm ON pres.hadm_id = adm.hadm_id;
"""

data = pd.read_sql_query(query, engine)

# Step 2: Data Preprocessing
# Simplify drug names and convert to one-hot encoding
data['drug'] = data['drug'].str.lower().str.strip()
data_onehot = pd.get_dummies(data[['hadm_id', 'drug']], columns=['drug'], prefix='', prefix_sep='')
data_onehot = data_onehot.groupby('hadm_id').max().reset_index()  # Max to ensure 1 if drug was given

# Merge with outcomes
outcomes = data[['hadm_id', 'hospital_expire_flag']].drop_duplicates()
features = data_onehot.merge(outcomes, on='hadm_id', how='left')

# Step 3: Feature and Target Preparation
X = features.drop(['hadm_id', 'hospital_expire_flag'], axis=1)
y = features['hospital_expire_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features for deep learning model
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Random Forest Classifier (Machine Learning Model)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Top 20 Feature Importances:")
for i in range(20):
    print(f"{X_train.columns[indices[i]]}: {importances[indices[i]]}")

# Evaluation
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"Random Forest - Accuracy: {accuracy_rf:.4f}, ROC AUC: {roc_auc_rf:.4f}")

# # Step 5: Deep Learning Model
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#     Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# # Evaluation
# y_pred_dl = (model.predict(X_test_scaled) > 0.5).astype("int32")
# accuracy_dl = accuracy_score(y_test, y_pred_dl)
# roc_auc_dl = roc_auc_score(y_test, y_pred_dl)
# print(f"Deep Learning - Accuracy: {accuracy_dl:.4f}, ROC AUC: {roc_auc_dl:.4f}")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).view(-1, 1)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, num_features):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(num_features, 64)
        self.layer2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

model = SimpleNN(X_train_scaled.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = []
    for data in test_loader:
        outputs = model(data[0])
        y_pred.extend(outputs.data.numpy())
    y_pred = np.array(y_pred).flatten()
    
y_pred_label = (y_pred > 0.5).astype(int)
accuracy_dl = accuracy_score(y_test, y_pred_label)
roc_auc_dl = roc_auc_score(y_test, y_pred)

print(f"Deep Learning - Accuracy: {accuracy_dl:.4f}, ROC AUC: {roc_auc_dl:.4f}")
