from flask import Flask, request, jsonify
from pydantic import BaseModel
from models import load_model
from utils import mask_pii
import torch
import json
import re
import traceback

# Initialize app and model
app = Flask(__name__)
model, tokenizer = load_model()

# Input schema (for documentation purposes - not used in Flask)
class EmailInput(BaseModel):
    email_body: str

def preprocess_email_text(text):
    """
    Preprocess email text by:
    1. Removing newlines
    2. Converting to lowercase
    3. Normalizing whitespace
    """
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route("/predict", methods=["POST"])
def classify_email():
    """
    API Endpoint to classify emails and return response JSON
    """
    try:
        raw_body = request.data.decode('utf-8', errors='replace')

        try:
            data = json.loads(raw_body)
            email_text = data.get('email_body', '')
        except json.JSONDecodeError:
            match = re.search(r'"email_body"\s*:\s*"(.*?)(?:"\s*\}|\s*",)', raw_body, re.DOTALL)
            if match:
                email_text = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            else:
                return jsonify({"error": "Could not parse email_body from request"}), 400

        if not email_text:
            return jsonify({"error": "email_body is required"}), 400

        preprocessed_text = preprocess_email_text(email_text)
        masked_text, entities = mask_pii(email_text)
        preprocessed_masked_text = preprocess_email_text(masked_text)

        inputs = tokenizer(preprocessed_masked_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        class_labels = ["Change", "Incident", "Problem", "Request"]
        category = class_labels[predicted_class]

        response = {
            "input_email_body": email_text,
            "list_of_masked_entities": [
                {
                    "position": entity["position"],
                    "classification": entity["classification"],
                    "entity": entity["entity"]
                } for entity in entities
            ],
            "masked_email": masked_text,
            "category_of_the_email": category
        }

        return jsonify(response)
    except Exception as e:
        error_detail = f"Error processing email: {str(e)}\n{traceback.format_exc()}"
        return jsonify({"error": error_detail}), 500

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Web form interface for browser-based testing
    """
    result = None
    if request.method == "POST":
        email_body = request.form.get("email_body", "")
        if email_body:
            preprocessed_text = preprocess_email_text(email_body)
            masked_text, entities = mask_pii(email_body)
            preprocessed_masked_text = preprocess_email_text(masked_text)
            inputs = tokenizer(preprocessed_masked_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_class = torch.argmax(outputs.logits, dim=1).item()
            class_labels = ["Change", "Incident", "Problem", "Request"]
            category = class_labels[predicted_class]
            result = {
                "input_email_body": email_body,
                "list_of_masked_entities": entities,
                "masked_email": masked_text,
                "category_of_the_email": category
            }

    return '''
        <h2>Email Classification System</h2>
        <form method="post">
            <textarea name="email_body" rows="10" cols="80" placeholder="Paste your email here..."></textarea><br>
            <input type="submit" value="Classify Email">
        </form>
    ''' + (f"<h3>Result:</h3><pre>{json.dumps(result, indent=2)}</pre>" if result else "")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
