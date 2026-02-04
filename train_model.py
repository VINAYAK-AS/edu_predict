import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

def train_and_save():
    # 1. Load the merged dataset
    df = pd.read_csv('data/keam_master_dataset.csv')
    
    # 2. Convert text to numbers (Label Encoding)
    categorical_cols = ['college_code', 'course', 'category']
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # 3. Define Features (X) and Target (y)
    X = df[['college_code', 'course', 'category', 'year', 'allotment_round']]
    y = df['last_rank']

    # 4. Train the Model
    print("Training Random Forest Model... Please wait 🌳")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 5. Save the Model and Encoders
    joblib.dump(model, 'models/keam_model.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    print("Model Training Complete. Files saved to the 'models' folder! 💾")

if __name__ == "__main__":
    train_and_save()