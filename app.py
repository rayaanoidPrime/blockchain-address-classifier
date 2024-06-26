from flask import Flask, jsonify
import joblib
from model import CryptoAddressFeatureExtractor  


app = Flask(__name__)

model = joblib.load('MLP_blockchain_address_classifier.joblib')
label_encoder = joblib.load('label_encoder.joblib')

@app.route('/' , methods=['GET'])
def health():
    return "Hello, World!"

@app.route('/predict/<address>', methods=['GET'])
def predict_blockchain(address):
    try:

        prediction = model.predict([address])[0]
        predicted_blockchain = label_encoder.inverse_transform([prediction])[0]
    
        return jsonify({
            'address': address,
            'predicted_blockchain': predicted_blockchain
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)