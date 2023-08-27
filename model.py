import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Load the merged dataset
df = pd.read_csv('merged_data11.csv')

# Split the dataset
X = df.drop(columns=['Value_class'])
y = df['Value_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize LabelEncoder
label_encoder = LabelEncoder()
label_encoder_y = LabelEncoder()

# Encode the categorical columns
X_train['Location'] = label_encoder.fit_transform(X_train['Location'])
X_test['Location'] = label_encoder.transform(X_test['Location'])
y_train = label_encoder_y.fit_transform(y_train)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform numerical features on the training set
XX_train_df_scaled = pd.DataFrame(scaler.fit_transform(X_train_resampled), columns=X_train.columns)

X_test_df_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Fit the LabelEncoder on both training and test targets
label_encoder_y.fit(y_train)
y_test_encoded = label_encoder_y.fit_transform(y_test)

LGB = lgb.LGBMClassifier(verbose=-1)

# Fit the model
LGB.fit(XX_train_df_scaled, y_train_resampled)
acc_train = LGB.score(XX_train_df_scaled, y_train_resampled)
acc_test = LGB.score(X_test_df_scaled, y_test_encoded)

print("Training Accuracy:", acc_train)
print("Test Accuracy:", acc_test)

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(LGB, X_test_df_scaled, y_test_encoded)


# Save the trained model
with open("model3.pkl", "wb") as f:
    pickle.dump(LGB, f)

print(classification_report(y_test_encoded, LGB.predict(X_test_df_scaled)))

print("Model saved as model2.pkl")