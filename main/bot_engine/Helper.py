import nltk
import pickle
import json

import tensorflow
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD

import numpy as np

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


class BotHelper:
    def __init__(self, data, modelPath):
        self.modelPath = modelPath
        self.data = data
        self.intents = self.loadData()

    def loadData(self):
        return json.loads(self.data)

    def processClassAndDoc(self):
        classes = []
        documents = []
        words = []
        for intent in self.intents['intents']:
            for pattern in intent['examples']:
                # print(pattern)
                # take each word and tokenize it
                w = nltk.word_tokenize(pattern['text'])
                words.extend(w)
                # adding documents
                documents.append((w, intent['title']))

                # adding classes to our class list
                if intent['title'] not in classes:
                    classes.append(intent['title'])
        return classes, documents, words
    
    def lemitize(self, words, ignore_words):
        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
        return words

    def saveAsPickle(self, fileName, data):
        pickle.dump(data,open(self.modelPath+fileName,'wb'))
    
    def prepateTrainData(self, words, classes, documents):

        # initializing training data
        training = []
        output_empty = [0] * len(classes)
        # print(output_empty)
        # print(documents)
        for doc in documents:
            # initializing bag of words
            bag = []
            # print(doc)
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            
            # lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            # create our bag of words array with 1, if word match found in current pattern
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)
            # print(bag)
            # output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training.append([bag, output_row])
        return training

class ANN:
    def __init__(self, input_shape, output_shape, modelPath):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.modelPath = modelPath
    
    def network(self):
        # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
        # equal to number of intents to predict output intent with softmax
        model = Sequential()
        model.add(Dense(128, input_shape=(self.input_shape,), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_shape, activation='softmax'))

        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def train(self, model, model_name, train_x, train_y, epochs):
        #fitting and saving the model
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=5, verbose=1)
        model.save(self.modelPath+model_name, hist)
