from bot_engine.intentHelper import BotHelper, ANN
import json
import random
import numpy as np
# from bot_engine.entityHelper import baseNER

def intentDetection():
    ignore_words = ['?', '!']
    
    data_file = open('/media/parth/DATA7/uroBot/input.json').read()
    modelPath = 'models/'
    helper = BotHelper(data_file, modelPath)
    helper.loadData()
    classes, documents, words = helper.processClassAndDoc()
    # print(classes) #intents
    # print(documents) #douments pairs e.i. (['how', 'are', 'you'], 'greetings'))
    # print(words) # all words

    words = helper.lemitize(words, ignore_words)
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))
    print (len(documents), "documents")
    print (len(classes), "classes", classes)
    print (len(words), "unique lemmatized words", words)

    helper.saveAsPickle('words.pkl', words)
    helper.saveAsPickle('classes.pkl', classes)
        
    training = helper.prepateTrainData(words, classes, documents)
    # print(training)

    # shuffle our features and turn into np.array
    random.shuffle(training)
    # print(training)
    training = np.array(training, dtype=object)
    # print(training)
    # create train and test lists. X - patterns, Y - intents
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    print("Training data created")


    input_shape = len(train_x[0])
    output_shape = len(train_y[0])
    neuralNetwork = ANN(input_shape, output_shape, modelPath)
    model = neuralNetwork.network()
    epochs = 200
    neuralNetwork.train(model, 'chatbot_model.h5', train_x, train_y, epochs)
    print("model created")




if __name__ == "__main__":
    intentDetection()

    


