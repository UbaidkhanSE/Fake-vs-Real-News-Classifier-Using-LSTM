# Fake vs Real News Classifier Using LSTM

## Objective
To develop a classifier that can distinguish between fake and real news using an LSTM (Long Short-Term Memory) neural network.

## Tasks

1. **Data Collection**
   - Obtained dataset with news articles labeled as fake or real.
   - Used publicly available datasets like the Fake News dataset from Kaggle.

2. **Pre-processing**
   - Cleaned the text data by removing special characters, stop-words, and performed tokenization and lemmatization.
   - Converted the text data into numerical form using techniques like TF-IDF and word embeddings (Word2Vec, GloVe).

3. **Model Development**
   - Implemented an LSTM neural network for the classification task.
   - Used Keras and TensorFlow for building and training the model.

4. **Model Evaluation**
   - Evaluated the model using metrics such as accuracy, precision, recall, and F1-score.
   - Plotted training and validation loss and accuracy over epochs.

5. **Visualization**
   - Created visualizations to represent findings, such as confusion matrix and ROC curves.

## Installation

To run this project, you'll need to install the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk gensim spacy tensorflow keras
