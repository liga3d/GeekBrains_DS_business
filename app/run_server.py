from models.text_preprocessor import TextPreprocessor

import numpy as np
import dill
import pandas as pd
dill._dill._reverse_typemap['ClassType'] = type

import flask

app = flask.Flask(__name__)
model = None
preprocessor = TextPreprocessor()

def load_model(model_path):
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)

@app.route("/", methods=["GET"])
def general():
	return "Welcome to toxic comments classification project"

@app.route("/predict", methods=["POST"])
def predict():

	data = {"success": False}

	if flask.request.method == "POST":
		comment = ""
		request_json = flask.request.get_json()
		if request_json["comment"]:
			comment = request_json['comment']
			try:
				comment = preprocessor.clean_text(comment)
				comment = preprocessor.lemmatization(comment)
				
				preds = model.predict(comment, debug=False)[:, 1][0]
				data["predictions"] = preds
				data["success"] = True
			except:
				pass

	return flask.jsonify(data)

if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	modelpath = "models/toxic_classifier.dill"
	load_model(modelpath)
	app.run()