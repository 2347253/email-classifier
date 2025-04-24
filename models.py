from transformers import BertTokenizer, BertForSequenceClassification

def load_model():
    model = BertForSequenceClassification.from_pretrained("saved_model")
    tokenizer = BertTokenizer.from_pretrained("saved_model")
    return model, tokenizer
