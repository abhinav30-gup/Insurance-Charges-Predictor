from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("ml_insurance.lb")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict_form():
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker = int(request.form['smoker'])  # 0 = no, 1 = yes
            region = int(request.form['region'])  # encoded region value

            input_data = ([[age, bmi, children, smoker, region]])
            prediction = float(model.predict(input_data)[0])
            predicted_cost = f"${prediction:,.2f}"

            return render_template('result.html', prediction=predicted_cost)
        except Exception as e:
            return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
