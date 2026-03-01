import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # <-- Added this import!
from sklearn.metrics import mean_absolute_error, r2_score
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

    # --- THE MISSING MAGIC LINE ---
    # Split the data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train the Model (Only on the 80% training data!)
    print("Training Random Forest Model... Please wait 🌳")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Make predictions on the 20% hidden test data
    y_pred = model.predict(X_test)

    # 6. Calculate the metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"R-squared Score: {r2 * 100:.2f}%")
    print(f"Mean Absolute Error (MAE): {mae:.2f} ranks")
    print("------------------------\n")

    # 7. Save the Model and Encoders
    joblib.dump(model, 'models/keam_model.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    print("Model Training Complete. Files saved to the 'models' folder! 💾")

if __name__ == "__main__":
    train_and_save()