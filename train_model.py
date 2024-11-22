import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


def train_and_save_model():
    # Read data
    df = pd.read_csv('weight-height.csv')
    df = df.drop('Gender', axis=1)

    # Prepare features and target
    x = df.drop('Weight', axis=1)  # Height is our feature
    y = df.drop('Height', axis=1)  # Weight is our target

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # Create and train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Save the model to disk
    joblib.dump(model, 'weight_prediction_model.joblib')
    print("Model trained and saved successfully!")

    # Optional: Print model performance
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(f"Training Score: {train_score:.4f}")
    print(f"Testing Score: {test_score:.4f}")


if __name__ == "__main__":
    train_and_save_model()