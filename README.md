# Review sentiment analyzer

The aim of the project was to create a sentiment analyzer for products review based on Amazon books reviews.


# 1. Download and prepare data

Data concerns reviews of books on Amazon and was downloaded from<br>
https://nijianmo.github.io/amazon/index.html

Due to the size of the dataset, it was sometimes necessary to use generators in preprocessing.

For a bigger contrast in sentiment, only reviews with an overall rating of 1.3 or 5 were selected. As 5-rated reviews made up nearly 80% of the collection, it was decided to undersampling to balance the grades. Then the missing data and duplicates were removed. In the next step, the target variable was created from the `overall` variable.

Then, the texts of the reviews were cleaned up. Removed:
* URL
* newline characters
* special signs
* numbers
* tags

Then, using `gensim.utils.simple_preprocess()`, further purification of the texts and tokenization were performed. Stopwords were removed using the `spacy` module and the `en_core_web_md` language model. There were used `pad_sequence` for padding - to prepare strings of equal length for each review.<br>
Finally, the dataset was splitted into train (75%) and test (25%).


# 2. Simple BILSTM and CNN models

To classify the sentiment of a given review, we will first use the bidirectional LSTM (BiLSTM) and Convolutional Neural Network (CNN). The neural networks were created using `tensorflow`.

For the data prepared in the previous script, we initiated each network with simple architecture. The BILSTM with only two layers: word embedding and  bidirectional LSTM layer and CNN with two convolution layers. Training took place over many sessions due to the long computation time.

Time of training one epoch on GPU with Google Colab:
* BILSTM - 40 minuts
* CNN - 2 minuts

# 3. Evaluation

Each model was evaluated based on accuracy, f-1 score and confusion matrix.

|    CNN   | Precision | Recall | f-1 score |
|:--------:|:---------:|:------:|:---------:|
| Negative |    0.75   |  0.79  |    0.77   |
|  Neutral |    0.69   |  0.64  |    0.67   |
| Positive |    0.78   |  0.79  |    0.78   |

|  BILSTM  | Precision | Recall | f-1 score |
|:--------:|:---------:|:------:|:---------:|
| Negative |    0.78   |  0.82  |    0.80   |
|  Neutral |    0.73   |  0.69  |    0.71   |
| Positive |    0.81   |  0.80  |    0.81   |

# 4. Conclusions

Based on the conducted experiments, we have the following conclusions:
* Both models give quite good results, about 75% accuracy.
* BILSTM gives slightly better results compared to CNN.
* CNN trains much faster than BILSTM (15 times faster).
* CNN requires regularization - it is more willing to overfit
* It was more difficult to create a model based on 5 rating categories - the differences in the sentiment of the reviews were blurred.
