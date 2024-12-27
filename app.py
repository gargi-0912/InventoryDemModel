from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

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

# Prepare the model
X = df[['festival_encoded', 'inventory_encoded']]
y = df['demand']

model = LinearRegression()
model.fit(X, y)

# Mapping dictionaries for encoding and decoding
festival_mapping = {name: code for code, name in enumerate(df['festival'].astype('category').cat.categories)}
inventory_mapping = {name: code for code, name in enumerate(df['inventory'].astype('category').cat.categories)}
reverse_inventory_mapping = {v: k for k, v in inventory_mapping.items()}

@app.route("/")
def home():
    """Render the home page with the prediction form."""
    return render_template("index.html", festivals=festival_mapping.keys(), inventories=inventory_mapping.keys())

@app.route("/predict", methods=["POST"])
def predict():
    """Handle the prediction request."""
    try:
        date = request.form.get("date")
        festival = request.form.get("festival")
        inventory = request.form.get("inventory")
        
        if not date or not festival or not inventory:
            return "All fields are required!", 400
        
        # Get encoded values
        festival_encoded = festival_mapping.get(festival, -1)
        inventory_encoded = inventory_mapping.get(inventory, -1)
        
        if festival_encoded == -1 or inventory_encoded == -1:
            return "Invalid input values!", 400

        # Predict demand
        prediction = model.predict([[festival_encoded, inventory_encoded]])[0]
        prediction = round(prediction, 2)

        return render_template(
            "result.html", 
            date=date, 
            festival=festival, 
            inventory=inventory, 
            predicted_demand=prediction
        )
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
