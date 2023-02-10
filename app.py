from flask import Flask, render_template, request
from models.message_classifier import MessageClassifier

app = Flask(__name__)
classifier = MessageClassifier()

classifier.load_model(
    model_path='./models/mlp_model.joblib', 
    json_classes_path='./models/classes.json', 
)

# global text cache 
text_cache = ""

category_map = {
    'Q': 'Question',
    'S': 'Schedule',
    'T': 'Teacher',
    'A': 'Assignment',
    'C': 'Common'
}

card_color_map = {
    'Q': '#ffff00',
    'S': '#00bfff',
    'T': '#0040ff',
    'A': '#8000ff',
    'C': '#00ff00'
}

@app.route('/', methods=['GET'])
def input_form():
    global text_cache
    text_cache = ""
    return render_template('input_form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    global text_cache
    text = request.form['text']

    if(text == text_cache):
        return render_template('input_form.html')
    text_cache= text

    # more readable output
    output = classifier.classify(text_input= text)
    card_color= card_color_map[output[0][-1]]
    output = [category_map[output[0][-1]], round(output[1], 3),]

    return render_template('input_form.html', message= text, output= output, card_color= card_color)


if __name__ == "__main__":
    app.run(debug=False)