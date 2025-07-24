from flask import Flask, render_template, request
import pickle
import numpy as np
import webbrowser
import threading

app = Flask(__name__)

# Load trained model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("symptom_encoder.pkl", "rb"))

# Use classes_ for MultiLabelBinarizer
all_symptoms = encoder.classes_

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")

        # Prepare input vector
        input_data = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
        input_data = np.array(input_data).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)[0]

    return render_template("index.html", symptoms=all_symptoms, prediction=prediction)

# Automatically open browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:500/")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(debug=True)
