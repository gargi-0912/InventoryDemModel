import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load historical data
data = {
    "date": ["2024-01-01", "2024-01-02", "2024-01-10", "2024-01-15", "2024-01-20", "2024-01-25", 
             "2024-02-01", "2024-02-10", "2024-02-15"],
    "demand": [100, 90, 110, 120, 80, 85, 95, 110, 100],
    "festival": ["New Year", "New Year", "Diwali", "Christmas", "Other", "Other", "New Year", "Diwali", "Other"],
    "inventory": ["Grocery", "Grocery", "Electronics", "Clothing", "Other", "Other", "Grocery", "Electronics", "Other"]
}
df = pd.DataFrame(data)

# Encode categorical data
df['festival_encoded'] = df['festival'].astype('category').cat.codes
df['inventory_encoded'] = df['inventory'].astype('category').cat.codes

# Prepare the features and target
X = df[['festival_encoded', 'inventory_encoded']]
y = df['demand']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Mapping dictionaries for encoding and decoding
festival_mapping = {name: code for code, name in enumerate(df['festival'].astype('category').cat.categories)}
inventory_mapping = {name: code for code, name in enumerate(df['inventory'].astype('category').cat.categories)}
reverse_inventory_mapping = {v: k for k, v in inventory_mapping.items()}

def predict_demand(date, festival, inventory):
    """
    Predicts the demand for a specific date, festival, and inventory type.
    
    Args:
        date (str): Date in YYYY-MM-DD format.
        festival (str): Festival name (e.g., "New Year", "Diwali", "Other").
        inventory (str): Inventory type (e.g., "Grocery", "Electronics").
    
    Returns:
        float: Predicted demand.
    """
    try:
        # Encode the inputs
        festival_encoded = festival_mapping.get(festival, -1)
        inventory_encoded = inventory_mapping.get(inventory, -1)

        # Validate inputs
        if festival_encoded == -1 or inventory_encoded == -1:
            raise ValueError("Invalid festival or inventory type!")

        # Make prediction
        prediction = model.predict([[festival_encoded, inventory_encoded]])[0]
        return round(prediction, 2)
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

if __name__ == "__main__":
    # Interactive user input for prediction
    print("Welcome to the Inventory Demand Prediction!")
    
    try:
        date = input("Enter the date (YYYY-MM-DD): ")
        festival = input("Enter the festival (New Year, Diwali, Christmas, Other): ")
        inventory = input("Enter the inventory type (Grocery, Electronics, Clothing, Other): ")

        predicted_demand = predict_demand(date, festival, inventory)
        print(f"Predicted demand for {inventory} during {festival} on {date}: {predicted_demand}")
    except ValueError as ve:
        print(ve)
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
