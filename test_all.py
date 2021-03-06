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

# Crea i modelli
model0_0 = keras.models.load_model('model0_0.h5')
model1_0 = keras.models.load_model('model1_0.h5')
model1_1 = keras.models.load_model('model1_1.h5')

words = []
for i in range(len(dicts)):
	words += [(w,i) for w in dicts[i]]
random.shuffle(words)

def predict(word):
	x = numpy.array([to_nn_input(word)])
	y = model0_0.predict(x)[0]
	res = [y]
	if y < 0.5:
		y = model1_0.predict(x)[0]
		res += [y]
		y = (0 if y < 0.5 else 1)
	else:
		y = model1_1.predict(x)[0]
		res += [y]
		y = (2 if y < 0.5 else 3)
	return (res, y)

def predict2(word):
    x = numpy.array([to_nn_input(word)])
    y0_0 = model0_0.predict(x)[0]
    y1_0 = model1_0.predict(x)[0]
    y1_1 = model1_1.predict(x)[0]
    en = (1-y0_0) * (1-y1_0)
    de = (1-y0_0) * (y1_0)
    fr = (y0_0) * (1-y1_1)
    it = (y0_0) * (y1_1)
    return [en,de,fr,it]

m = numpy.arange(0.30, 1.00, 0.01) # Confidence threshold
m = numpy.append(m, [0.992,0.994,0.996,0.998,0.999])
covered = [0]*len(m)
correct = [0]*len(m)
tot_examples = float(len(words))
examples_processed = 0
for (word, index) in words:
    nn_x = numpy.array([to_nn_input(word)])
    pred_y = predict2(word)
    pred_index = arg_max(pred_y)
    pred_confidence = pred_y[pred_index]
    prediction_is_correct = pred_index == index or word in dicts[pred_index]
    for i in range(len(m)):
        if pred_confidence >= m[i]:
            covered[i] += 1
            if prediction_is_correct:
                correct[i] += 1
        else:
            break
    examples_processed += 1
    if examples_processed%1000 == 0:
        print('Processed: {0}/{1}'.format(examples_processed, len(words)))
if examples_processed % 1000 != 0:
    print('Processed: {0}/{0}'.format(len(words)))

for i in range(len(m)):
    print('{0:.2f} & {1} & {2:.6f} & {3} & {4:.6f} & {5:.6f}'.format(m[i], covered[i], covered[i]/tot_examples, correct[i], correct[i]/float(covered[i]), correct[i]/tot_examples))

plot1_x = numpy.array(m)
plot1_y1 = numpy.array(covered)
plot1_y2 = numpy.array(correct)

plt.figure(1)
plt.xlabel('Min Confidence')
plt.ylabel('# of Examples')
plt.xlim(0.3,1.0)
plt.ylim(0,len(words))
plt.plot(plot1_x, plot1_y1, color = 'blue', linestyle = '-', label = 'Covered')
plt.plot(plot1_x, plot1_y2, color = 'green', linestyle = '-', label = 'Correct')
plt.legend(loc = 'best')
plt.savefig('cov_acc_1_all.png')

plot2_x = plot1_x
plot2_y1 = numpy.array([c/tot_examples for c in covered])
plot2_y2 = numpy.array([c/tot_examples for c in correct])
plot2_y3 = numpy.array([correct[i]/float(covered[i]) for i in range(len(m))])

plt.figure(2)
plt.xlabel('Min Confidence')
plt.ylabel('% of Examples')
plt.xlim(0.3,1.0)
plt.ylim(0.0,1.0)
plt.plot(plot2_x, plot2_y1, color = 'blue', linestyle = '-', label = 'Covered')
plt.plot(plot2_x, plot2_y2, color = 'green', linestyle = '-', label = 'Correct / Total')
plt.plot(plot2_x, plot2_y3, color = 'red', linestyle = '-', label = 'Correct / Covered')
plt.legend(loc = 'best')
plt.savefig('cov_acc_2_all.png')