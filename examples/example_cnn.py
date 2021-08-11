import numpy as np; np.random.seed(121);
from matplotlib import pyplot as plt

import phidnet   # Import phidnet



############################################### Data
X, T, X_test, T_test = phidnet.datasets.mnist.load_2d()
X, X_test = X / 255, X_test / 255
T = phidnet.one_hot_encode.encode_array(T, length=10)
print("input shape:", X.shape)
print("output shape:", T.shape)
print("test input shape:", X_test.shape)
print("test output shape:", T_test.shape)
###############################################



###############################################
def create_linear_model():

    model = phidnet.CNN.convolution.CNN()

    model.add_layer(phidnet.CNN.layer.Linear(784, 500))
    model.add_layer(phidnet.CNN.layer.Relu())
    model.add_layer(phidnet.CNN.layer.Linear(500, 400))
    model.add_layer(phidnet.CNN.layer.Relu())
    model.add_layer(phidnet.CNN.layer.Linear(400, 300))
    model.add_layer(phidnet.CNN.layer.Relu())
    model.add_layer(phidnet.CNN.layer.Linear(300, 200))
    model.add_layer(phidnet.CNN.layer.Relu())
    model.add_layer(phidnet.CNN.layer.Linear(200, 100))
    model.add_layer(phidnet.CNN.layer.Relu())
    model.add_layer(phidnet.CNN.layer.Linear(100, 10))
    model.add_layer(phidnet.CNN.layer.Relu())
    model.add_layer(phidnet.CNN.layer.Output())
    model.set_optimizer(phidnet.CNN.optimizer.Adam(0.0001))
    return model




model = create_linear_model()
X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3]))
model.fit(X, T, epochs=20, plot=True, batch_size=10000, print_rate=1, val_x=X_test, val_y=T_test, val_size=20)




number = 111
predicted = model.predict(X_test[number])
print('predict:', predicted)
print('predict:', np.argmax(predicted))
print("right answer:", T_test[number])
img = X_test[number].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()
###############################################