import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tg import expose, TGController, AppConfig
from wsgiref.simple_server import make_server

model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])

def train(x_train, y_train):
    x_train = np.array([x_train])
    y_train = [int(y_train)]
    y_train = np.array(y_train)
    model.fit(x_train, y_train,
          epochs=20,
          batch_size=1)
def test(x_test):
    score = model.predict(x_test, batch_size=1)
    return '[' + ','.join(str(e) for e in score) + ']'



class RootController(TGController):
    @expose()
    def test(self, vector):
        vector = eval(vector)
        return test(np.array([vector]))
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
