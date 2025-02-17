import os
import uuid
import urllib.request
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import google.generativeai as genai
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Load the model
with open('model.json', 'r') as j_file:
    loaded_json_model = j_file.read()
model = model_from_json(loaded_json_model)
model.load_weights('model.h5')

# Allowed image extensions
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'jfif'}

# Class labels
classes = [
    'Actinic Keratoses',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevi',
    'Vascular naevus'
]

# Function to check if file is allowed


def allowed_file(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXT

# Prediction function


def predict(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32') / 255.0

    result = model.predict(img)[0]

    # Create a dictionary mapping probabilities to class labels
    dict_result = {prob: cls for prob, cls in zip(result, classes)}

    # Sort results in descending order
    sorted_probs = sorted(result, reverse=True)[:3]

    # Get the top 3 classes and probabilities
    class_result = [dict_result[prob] for prob in sorted_probs]
    prob_result = [round(prob * 100, 2) for prob in sorted_probs]

    return class_result, prob_result

# Home route


@app.route('/')
def home():
    return render_template("index.html",API_KEY=API_KEY)

# Success route for predictions


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')

    if request.method == 'POST':
        # Handling image upload via URL
        if request.form.get('link'):
            link = request.form.get('link')
            try:
                unique_filename = f"{uuid.uuid4()}.jpg"
                img_path = os.path.join(target_img, unique_filename)
                urllib.request.urlretrieve(link, img_path)
                img = unique_filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accessible or input is inappropriate.'

        # Handling image upload via file input
        elif request.files.get('file'):
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = file.filename
                img_path = os.path.join(target_img, filename)
                file.save(img_path)
                img = filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            else:
                error = "Please upload images with jpg, jpeg, or png extensions only."

        if error:
            return render_template('index.html', error=error)
        return render_template(
            'success.html',
            img=img,
            predictions=predictions)

    return render_template('index.html')


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
