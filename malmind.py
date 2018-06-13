import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from tg import expose, TGController, AppConfig
from wsgiref.simple_server import make_server
import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Neural network classification service.')
parser.add_argument('-E')
parser.add_argument('-G')
parser.add_argument('-M')
args = parser.parse_args()

args.M = os.environ['MODEL'] if args.M == None else args.M
if args.G != None:
    x_train = np.random.random((100, 100))
    y_train = np.random.randint(1, size=(100, 1))

samples_memory = []
outputs_memory = []

if args.M == None:
    model = Sequential()
    model.add(Dense(64, input_dim=100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
else:
    json_file = open(args.M, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(args.M + '.h5')

model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])
if args.G != None:
    history = model.fit(x_train, y_train, epochs=int(args.E) if args.E else 30, batch_size=100)

    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def train(x_train, y_train):
    samples_memory.append(x_train)
    x_train = np.array(samples_memory)
    outputs_memory.append([int(y_train)])
    y_train = np.array(outputs_memory)
    model.fit(x_train, y_train,
          epochs=int(args.E) if args.E else 30,
          batch_size=len(x_train))
def test(x_test):
    score = model.predict(x_test, batch_size=1)
    print score
    return '[' + ','.join(str(e) for e in score[0]) + ']'
def save(name):
    model_json = model.to_json()
    with open(name, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(name + '.h5')



class RootController(TGController):
    @expose()
    def test(self, vector):
        vector = eval(vector)
        return test(np.array([vector]))
    @expose()
    def save(self, name):
        save(name)
        return 'ok'
    @expose()
    def train(self, vector, malwareClass):
        vector = eval(vector)
        train(vector, malwareClass)
        return 'ok'
config = AppConfig(minimal=True, root_controller=RootController())

application = config.make_wsgi_app()
print("Serving on port 3008...")
httpd = make_server('', 3008, application)
httpd.serve_forever()
