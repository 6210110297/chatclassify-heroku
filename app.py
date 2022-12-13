from flask import Flask, jsonify, render_template, request
from models.message_classifier import MessageClassifier

app = Flask(__name__)
classifier = MessageClassifier()

classifier.load_model(model_path='./models/rf_model.joblib', json_classes_path='./models/classes.json')

# global text cache 
# !!! dont use when already have DB
text_cache = ""

category_map = {
    'Q': 'Question',
    'S': 'Schedule',
    'T': 'Teacher',
    'A': 'Assignment',
    'C': 'Common'
}

@app.route('/')
def hello():
    return "Hello Chat"

@app.route('/form', methods=['GET'])
def input_form():
    global text_cache
    text_cache = ""
    return render_template('input_form.html')

@app.route('/form', methods=['POST'])
def my_form_post():
    global text_cache
    text = request.form['text']

    if(text == text_cache):
        return render_template('input_form.html')
    text_cache= text

    # more readable output
    output = classifier.classify(text_input= text)
    output = [category_map[output[0][-1]], round(output[1], 3)]

    return render_template('input_form.html', message= text, output= output)


if __name__ == "__main__":
    app.run(debug=False)