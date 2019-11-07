# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
# import tensorflow as tf

# from tensorflow.keras.applications import ResNet50
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np
import io
import os
import flask


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

top_model_path = 'model.h5'
top_model_weights_path = 'weights.h5'

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	# model = ResNet50(weights="imagenet")
	model = tensorflow.keras.models.load_model(top_model_path)	
	# model.load_weights(top_model_weights_path, by_name=False)
	model._make_predict_function() 

def prepare_image(image, target):
	# image = load_img(file_path, target)
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image /= 255 
	app.logger.info("test for CI/CD")
	app.logger.info("test for CI/CD05")
	# return the processed image
	return image

def model_predict(image):
    describe = []
    prediction = model.predict(image)
    app.logger.info("prediction np array: {}".format(prediction))
    if prediction < 0.5:
        describe.append('cat %.2f%%' % (100 -prediction*100))
    else:
        describe.append('dog %.2f%%' % (prediction*100))
    
    return describe


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	# data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		print("get image")
		# # read the image in PIL format
		# image = flak.request.files["image"].read()
		# image = Image.open(io.BytesIO(image))
		# Get the file from post request
		image = Image.open(request.files['file'].stream)
		print("get image")
		# Save the file to ./uploads
		# basepath = os.path.dirname(__file__)
		# file_path = os.path.join(
		# 	basepath, 'uploads', secure_filename(f.filename))
		# f.save(file_path)

		# preprocess the image and prepare it for classification
		image = prepare_image(image, target=(150, 150))
		print("prepare image")
		# classify the input image and then initialize the list
		# of predictions to return to the client
		preds = model_predict(image)
		result = str(preds[0])
		print(result)
		return result
	# return the data dictionary as a JSON response
	# return flask.jsonify(result)
	return None

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run(port=5000, debug=True, host='0.0.0.0')