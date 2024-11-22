from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model_file = "mall_customers.pkl"
with open(model_file, 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        features = [
            request.form.get('Annual Income (k$)'),
            request.form.get('Spending Score (1-100)')
        ]
        input_features = np.array(features).reshape(1, -1)
        prediction = model.predict(input_features)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)