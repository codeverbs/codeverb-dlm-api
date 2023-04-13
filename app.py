import torch
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from architecture.Utils import set_env, set_seed, print_time, load_model, create_custom_gpt2_tokenizer, Inference, final_processing

app = Flask(__name__)

# available models
available_models = [
    "CodeVerbDLM-0.3B"
]

# inference types
inference_types = [
    "Comment2Python",
    # "Algo2Python",
]

# Load our model
set_env()
set_seed(42, deterministic=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading Model on Device: ", device)
# ckpt = "./model"
with print_time('Loading Parameters: '):
    model = load_model(ckpt="./model/", fp16=True).to(device)
with print_time('Fetching Tokenizer: '):
    tokenizer = create_custom_gpt2_tokenizer()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = 50256


# Define a simple route
@app.route('/', methods=['GET'])
@cross_origin()
def home():
    msg = {
        "API Name": "CodeVerb DLM API",
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
        if inference_type not in inference_types:
            msg = {
                "error": "Inference type not available! Available inference types: {}".format(inference_types)
            }
            return jsonify(msg), 400, {'Content-Type': 'application/json; charset=utf-8'}
        if model_name not in available_models:
            msg = {
                "error": "Model not available! Available models: {}".format(available_models)
            }
            return jsonify(msg), 400, {'Content-Type': 'application/json; charset=utf-8'}
        
        # Predicted code here
        completion = Inference(device=device, model=model, tokenizer=tokenizer, context=query, pad_token_id=50256, num_return_sequences=1, temp=0.2, top_p=0.95, max_length_sample=1024)[0]
        predicted_code = final_processing(completion)
        msg = {
            "query": query,
            "result": predicted_code
        }
        return jsonify(msg), 200, {'Content-Type': 'application/json; charset=utf-8'}


if __name__ == '__main__':
    app.run(port=5025,debug=True)