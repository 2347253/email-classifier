# Email Classification and PII Masking API

This project provides a RESTful API for:
- Classifying support emails into categories (`Incident`, `Request`, `Problem`, `Change`)
- Masking personally identifiable information (PII) like names, emails, phone numbers, and account numbers
- Returning both masked email content and masked entities
- Built using FastAPI and deployed on Hugging Face Spaces

---

## Features

- ✅ PII masking (regex + spaCy)
- ✅ Multilingual email support
- ✅ BERT-based email classification
- ✅ API accepts POST requests with email body
- ✅ JSON response with masked text, original entities, and classification label

---

## API Endpoint

`POST /predict`

### Request Body:
```json
{
  "email": "Hi, my name is John Smith. My account number is 123456. Please help."
}
```
### Response:
```
{
  "masked_email": "Hi, my name is <NAME>. My account number is <ACCOUNT_NUM>. Please help.",
  "entities": [
    {"entity": "John Smith", "type": "NAME", "start": 17, "end": 27},
    {"entity": "123456", "type": "ACCOUNT_NUM", "start": 50, "end": 56}
  ],
  "classification": "Request"
}
```

---

## Installation

1. Clone the repo:
```
git clone https://github.com/your-username/email-api
cd email-api
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run Local:
```
python app.py
```

---

## Deployment

This API is deployed on Hugging Face Spaces.
