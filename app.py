from flask import Flask, jsonify, render_template, request
from models.message_classifier import MessageClassifier

app = Flask(__name__)
classifier = MessageClassifier()

classifier.load_model(model_path='./models/rf_model.joblib', json_classes_path='./models/classes.json')

# global text cache 
# !!! dont use when already have DB
text_cache = ""

data = [
        {
            "id": 1,
            "library": "Pandas",
            "language": "Python"
        },
        {
            "id": 2,
            "library": "requests",
            "language": "Python"
        },
        {
            "id": 3,
            "library": "NumPy",
            "language": "Python"
        }
    ]

@app.route('/')
def hello():
    return "Hello Chat"


@app.route('/api', methods=['GET'])
def get_api():
    return jsonify(data)

@app.route('/form')
def input_form():
    return render_template('input_form.html')

@app.route('/form', methods=['POST'])
def my_form_post():
    global text_cache
    text = request.form['text']

    if(text == text_cache):
        return render_template('input_form.html')
    text_cache= text

    output = classifier.classify(text_input= text)
    return render_template('input_form.html', message= text, output= output)


if __name__ == "__main__":
    app.run(debug=False)