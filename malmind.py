import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tg import expose, TGController, AppConfig
from wsgiref.simple_server import make_server

model = Sequential()
model.add(Dense(64, input_dim=100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

def train(x_train, isMalware):
    isMalware = int(isMalware)
    y_train = [1] if isMalware else [0]
    y_train = np.array(y_train);
    model.fit(x_train, y_train,
          epochs=20,
          batch_size=1)
def test(x_test):
    y_test = np.array([1])
    score = model.evaluate(x_test, y_test, batch_size=1)
    return '[' + ','.join(str(e) for e in score) + ']'



class RootController(TGController):
    @expose()
    def test(self, vector):
        vector = eval(vector);
        return test(np.array([vector]))
config = AppConfig(minimal=True, root_controller=RootController())

application = config.make_wsgi_app()
print("Serving on port 3008...")
httpd = make_server('', 3008, application)
httpd.serve_forever()
