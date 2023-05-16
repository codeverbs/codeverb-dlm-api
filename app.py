import torch
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from transformers import T5ForConditionalGeneration, AutoTokenizer
from utils import *

app = Flask(__name__)

# available models
available_models = [
    "CodeVerbTLM-0.7B"
]

# inference types
inference_types = [
    "Comment2Python"
    "Speech2Python",
    "Algo2Python",
]

# Load our model
checkpoint = "Salesforce/codet5p-770m-py"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading Model on Device: ", device)
with print_time('Loading Parameters: '):
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
with print_time('Fetching Tokenizer: '):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Define a simple route
@app.route('/', methods=['GET'])
@cross_origin()
def home():
    msg = {
        "API Name": "CodeVerb TLM 0.7M API",
        "API Version": "v1.0",
        "API Status": "Running",
        "Available Models": available_models
    }
    return jsonify(msg), 200, {'Content-Type': 'application/json; charset=utf-8'}


# Define a route that accepts POST requests with JSON data
@app.route('/api/predict', methods=['POST'])
@cross_origin()
def process_data():
    if request.method == 'POST':
        data = request.json
        query = data['query']
        model_name = data['model']
        inference_type = data['inference_type']
        # if inference_type not in inference_types:
        #     msg = {
        #         "error": "Inference type not available! Available inference types: {}".format(inference_types)
        #     }
        #     return jsonify(msg), 400, {'Content-Type': 'application/json; charset=utf-8'}
        if model_name not in available_models:
            msg = {
                "error": "Model not available! Available models: {}".format(available_models)
            }
            return jsonify(msg), 400, {'Content-Type': 'application/json; charset=utf-8'}
        
        # Preprocess input
        query = preprocess_string(query)
        # Predicted code here
        input = tokenizer.encode(query, return_tensors="pt").to(device)
        predicted_code = model.generate(input, max_length=512)
        msg = {
            "query": query,
            "result": predicted_code
        }
        return jsonify(msg), 200, {'Content-Type': 'application/json; charset=utf-8'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083, debug=False)