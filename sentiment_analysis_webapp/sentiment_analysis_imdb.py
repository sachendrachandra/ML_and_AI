#Import Libraries
import numpy
from numpy import array
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
import re
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

# Load the dataset but only keep the top n words and zero out the rest i.e keep vocabulary size as 5000
top_words = 5000 #vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

'''We can use the dictionary returned by imdb.get_word_index() to map the review back to the original words.'''
word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}


max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

'''Remember that our input is a sequence of words (technically, integer word IDs) of maximum length = max_review_length, and our output is a binary sentiment 
label (0 or 1).
'''
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=64)

#Calculate Accuracy
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#Run these codes first in order to install the necessary libraries and perform authorization
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}



#Now save the model in required directory
model.save('sentiment_analysis_model_new.h5')
print("Saved model to disk")

#Code to load the saved model
model = load_model('/content/drive/Sentiment_Analysis/sentiment_analysis_model_new.h5')
print("Model Loaded")





















