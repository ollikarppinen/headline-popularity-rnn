#encoding: utf-8

import pandas
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.data_utils import get_file
import numpy
from sklearn.metrics import confusion_matrix

def fetch_csv():
    csv = pandas.read_csv("http://207.154.192.240/ampparit/ampparit.csv")
    csv['title'] = csv['title'].str.lower()
    return csv

data = fetch_csv()
sorted_data = data.sort_values(['clicks'])

rows = len(sorted_data)

epochs = 10
ratio = 10
tops = sorted_data[(rows - rows // ratio):]
bottoms = sorted_data[:(rows // ratio)]

tops['label'] = 1
bottoms['label'] = 0

labeled_data = pandas.concat([tops, bottoms])

titles = labeled_data['title']
labels = labeled_data['label']

chars = {None: 0}

for title in titles:
    for char in title:
        if char in chars:
            chars[char] += 1
        else:
            chars[char] = 1

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

title_max_len = 128

X = []
y = numpy.array(labels)

for i, title in enumerate(titles):
    X.append([])
    for j, char in enumerate(title[-title_max_len:]):
        X[i].append(char_indices[char])

X = sequence.pad_sequences(X, maxlen=title_max_len)

ids = numpy.arange(len(X))
numpy.random.shuffle(ids)

# shuffle
X = X[ids]
y = y[ids]

data_set_count = len(X) // 1
test_set_count = data_set_count // 5
test_set_start_index = data_set_count - test_set_count

X_train = X[:test_set_start_index]
X_test = X[test_set_start_index:data_set_count]

y_train = y[:test_set_start_index]
y_test = y[test_set_start_index:data_set_count]
print("Training data positive labels: %.2f%%" % (sum(y_train) / len(y_train) * 100))
print("Testing data positive labels: %.2f%%" % (sum(y_test) / len(y_test) * 100))

model = Sequential()
model.add(Embedding(len(chars), 1, input_length=title_max_len))
model.add(LSTM(10))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history_callback = model.fit(X_train, y_train, nb_epoch=epochs)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

y_predict = model.predict_classes(X_test)
conf_matrix = confusion_matrix(y_test, y_predict)
print("Confusion matrix: ")
print(conf_matrix)
