from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=False)