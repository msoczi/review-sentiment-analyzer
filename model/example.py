# -----------------------------------------------------
# --- EXAMPLE OF MODEL USE FOR SENTIMENT PREDICTION ---
# -----------------------------------------------------

print('Reading data...')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.preprocessing.sequence import pad_sequences
from pickle import load
from tensorflow.keras.models import load_model

# Read model
model_BILSTM = load_model('BILSTM_model')

# Read tokenizer
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = load(f)

with open('example_review.txt', 'r') as fl:
    example_review = fl.read()

def predict_sentiment(text):
    """
    Funciton predict sentiment of review and returns it.
    
    Arguments:
    text (string) - review text
    
    Returns:
    overall_pred (string) - predicted sentiment
    
    """
    overall = ["NRGATIVE", "NEUTRAL", "POSITIVE"]
    sequence = tokenizer.texts_to_sequences([text])
    test_review = pad_sequences(sequence, maxlen=600)
    overall_pred = overall[model_BILSTM.predict(test_review).argmax(axis=1)[0]]
    return overall_pred


print('Text review:')
print(example_review)
print('-------------------------------------------------\n')
print('Predict sentiment...\n')
print('Sentiment:', predict_sentiment(example_review))
