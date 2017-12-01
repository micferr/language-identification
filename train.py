import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.utils import plot_model

import numpy

import matplotlib.pyplot as plt

import random
import sys
import timeit

word_length = 35 # Lunghezza massima di una parola (caratteri in eccesso ignorati)
epochs = 50 # Numero di epoch per la fase di training
batch_size = 1024 # Dimensione della minibatch per SGD
validation_ratio = 0.1 # Percentuale di esempi da usare per calcolare l'errore di generalizzazione
letters_in_alphabet = 26
onehot_vector_length = word_length * letters_in_alphabet
dict_files = ['en-us.txt', 'de-de.txt', 'fr-fr.txt', 'it-it.txt'] # File dizionario

''' 
Carica un file dizionario e ritorna un array di parole.
Il formato e' di una parola per riga.
'''
def load_dict(f):
    with open(f) as file:
        c = file.readlines()
    d = [line.strip() for line in c]
    return d

'''
Converte una parola nel formato di input per la rete neurale
'''
def to_nn_input(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    alphabet = {}
    for c in range(len(letters)):
        alphabet[letters[c]] = c + 1
    arr = [0]*word_length
    word = word.lower()
    for i in range(min(len(word),word_length)):
        c = word[i]
        if c in alphabet:
            v = alphabet[c]
            arr[i] = v
    return arr 

def arg_max(arr):
    if len(arr) < 1:
        raise ValueError('empty array')
    max_value = arr[0]
    max_index = 0
    for i in range(len(arr)):
        if arr[i] > max_value:
            max_value = arr[i]
            max_index = i
    return max_index

# Carica i dizionari 
dicts = []
for i in range(len(dict_files)):
    file = dict_files[i]
    x = load_dict(file)
    dicts += [ x ]

def train(model, trainData):
    history = []
    for e in range(epochs):
        start_time = timeit.default_timer()
        random.shuffle(trainData)
        i = 0
        n = 0
        tot_metrics = [0,0] # [loss, accuracy]
        print("Epoch {0}".format(e+1))
        while i + batch_size < len(trainData):
            X = numpy.array([to_nn_input(row[0]) for row in trainData[i:i+batch_size]])
            Y = [row[1] for row in trainData[i:i+batch_size]]
            metrics = model.train_on_batch(X,Y)
            i += batch_size
            n += 1
            tot_metrics[0] += metrics[0]
            tot_metrics[1] += metrics[1]
            sys.stdout.write("\rProcessed: {0}/{1}. Elapsed: {2}".format(i, len(trainData), (timeit.default_timer()-start_time)))
            sys.stdout.flush()
        print("")
        loss, acc = tot_metrics[0]/n, tot_metrics[1]/n
        history += [(e, loss, acc)]
    return history

# Crea i modelli
models = []
model_names = ['model0_0','model1_0', 'model1_1']
for i in range(3):
    if i == 0:
        trainData = [(d,[0]) for d in (dicts[0]+dicts[1])] + [(d,[1]) for d in dicts[2]+dicts[3]]
    elif i == 1:
        trainData = [(d,[0]) for d in dicts[0]] + [(d,[1]) for d in dicts[1]]
    else:
        trainData = [(d,[0]) for d in dicts[2]] + [(d,[1]) for d in dicts[3]]
    random.shuffle(trainData)
    trainData = trainData[:int(len(trainData)*0.2)]

    model = Sequential()
    model.add(Embedding(27, 100, input_length = word_length))
    model.add(LSTM(64, return_sequences = True))
    model.add(LSTM(32, return_sequences = True))
    model.add(LSTM(32))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(
        loss = keras.losses.binary_crossentropy,
        optimizer = keras.optimizers.RMSprop(lr = 0.01),
        metrics = ['accuracy']
    )

    plot_model(model, to_file = model_names[i]+".png", show_shapes = True)

    train(model, trainData)

    model.save(model_names[i]+".h5")
    models += [model]
