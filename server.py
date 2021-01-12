import numpy as numpy
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load(open('models/pipe_clf_chkpt.joblib', 'rb'))
model_clf = model['pipeline_clf']

@app.route('/api', methods = ['POST'])
def predict():
	data = request.get_json(force = True)
	prediction = model_clf.predict(data["X"])
	output_text = 'Text:' + str(data["X"])
	output = 'Class:' + str(prediction)

	return jsonify(output_text, output)

if __name__ == '__main__':
	app.run(port = 8080, debug = True)