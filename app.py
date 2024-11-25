import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib

from flask import Flask, render_template, jsonify, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from keras.models import load_model
import joblib

app = Flask(__name__)

# Load trained models
lstm_model = load_model('models/lstm_model.h5')
rf_model = joblib.load('models/random_forest_model.pkl')

# Generate a plot for LSTM predictions and Random Forest predictions
def generate_plot(stock_names, lstm_preds, rf_pred):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot LSTM predictions for each stock
    ax.bar(stock_names, lstm_preds, color='blue', alpha=0.7, label='LSTM Predictions')

    # Add Random Forest overall prediction as a line
    ax.axhline(y=rf_pred, color='red', linestyle='--', label='Portfolio Prediction')

    # Title and labels
    ax.set_title('Stock Predictions and Portfolio Outlook')
    ax.set_xlabel('Stocks')
    ax.set_ylabel('Predicted Returns')
    ax.legend()

    # Save the plot as a base64-encoded PNG
    img = io.BytesIO()
    plt.savefig(img, format='png')  # Save plot to a byte stream
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')  # Convert to base64 string
    plt.close(fig)  # Close the figure to avoid GUI interaction
    return plot_url

# Generate insights based on predictions
def generate_insights(rf_pred, stocks, lstm_predictions):
    # General message based on portfolio prediction
    if rf_pred > 0.6:
        outlook = "strong positive"
    elif rf_pred > 0.3:
        outlook = "moderate"
    else:
        outlook = "cautious"
    
    # Specific advice for each stock based on its prediction
    stock_advice = []
    for i, stock in enumerate(stocks):
        predicted_return = lstm_predictions[i]
        if predicted_return > 0.5:
            stock_advice.append(f"Consider increasing your investment in {stock['name']} as it has a predicted return of {predicted_return:.2f}.")
        elif predicted_return == 0:
            stock_advice.append(f"Consider reviewing your position in {stock['name']}. Its predicted return is neutral (0.00).")
        else:
            stock_advice.append(f"Be cautious with {stock['name']}. Its predicted return is low ({predicted_return:.2f}).")

    # Generate the final insights message
    if outlook == "strong positive":
        advice = f"The model suggests a {outlook} outlook for your portfolio. You may want to increase exposure to high-performing stocks like {stocks[0]['name']}."
    elif outlook == "moderate":
        advice = "The model suggests a moderate outlook. Consider maintaining your current investments, but keep an eye on the market."
    else:
        advice = "The model suggests a cautious outlook. You might want to consider diversifying into safer investments or reducing exposure to riskier stocks."

    return {"outlook": advice, "stock_advice": stock_advice}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user data from the request
        data = request.get_json()
        stocks = data.get('stocks', [])  # List of stocks with names and amounts

        # Simulate input data for predictions
        test_data = np.random.rand(60, 3).reshape(1, 60, 3)  # Placeholder for time-series data

        # LSTM model predictions
        lstm_predictions = lstm_model.predict(test_data).flatten()

        # Ensure LSTM predictions match the number of stocks
        if len(stocks) > len(lstm_predictions):
            lstm_predictions = np.pad(lstm_predictions, (0, len(stocks) - len(lstm_predictions)), 'constant', constant_values=0)

        # Prepare input for Random Forest (ensure exactly 5 features)
        rf_input = lstm_predictions[:5]  # Use the first 5 features if enough are available
        if len(rf_input) < 5:
            rf_input = np.pad(rf_input, (0, 5 - len(rf_input)), 'constant', constant_values=0)  # Pad with zeros

        # Random Forest overall portfolio prediction
        rf_prediction = rf_model.predict(rf_input.reshape(1, -1))[0]

        # Convert NumPy array or float32 values to Python native types
        lstm_predictions = lstm_predictions.astype(float)  # Ensure float32 values are converted to float
        rf_prediction = float(rf_prediction)  # Convert to native Python float

        # Generate plot and insights
        stock_names = [stock["name"] for stock in stocks[:len(lstm_predictions)]]
        plot_url = generate_plot(stock_names, lstm_predictions[:len(stock_names)], rf_prediction)
        
        # Pass both 'stocks' and 'lstm_predictions' to generate_insights
        insights = generate_insights(rf_prediction, stocks, lstm_predictions)

        # Match predictions to user stocks
        matched_predictions = [
            {"stock": stock["name"], "predicted_return": lstm_predictions[i]} 
            for i, stock in enumerate(stocks[:len(lstm_predictions)])
        ]

        return jsonify({
            'lstm_predictions': lstm_predictions.tolist(),  # Convert to Python list
            'rf_prediction': rf_prediction,  # Convert to native float
            'plot_url': plot_url,
            'insights': insights['outlook'],
            'stock_advice': insights['stock_advice'],
            'matched_predictions': matched_predictions
        })
    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
