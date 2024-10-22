from flask import Flask, request, jsonify, send_from_directory
from intent_recognition.intent_model import IntentRecognizer

app = Flask(__name__)

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/detect-intent', methods=['POST'])
def detect_intent_route():
    data = request.json
    user_input = data.get('user_input')

    if not user_input:
        return jsonify({'error': 'User input is required'}), 400

    intent_recognizer = IntentRecognizer()
    detected_intent, similarity_score = intent_recognizer.classify_intent(user_input)

    response_message = f"I detected your intent as '{detected_intent}' with a similarity score of {similarity_score:.2f}."

    return jsonify({
        'response': response_message,
    })



if __name__ == '__main__':
    app.run(debug=True)
