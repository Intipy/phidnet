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

    model.add_layer(phidnet.CNN.layer.Conv(1, 1, kernel_shape=(3, 3), stride=(1, 1), padding=False))
    model.add_layer(phidnet.CNN.layer.Relu())
    model.add_layer(phidnet.CNN.layer.Max_Pooling(shape=(2, 2), stride=(2, 2)))

    model.add_layer(phidnet.CNN.layer.Flatten())

    model.add_layer(phidnet.CNN.layer.Linear(196, 100))
    model.add_layer(phidnet.CNN.layer.Relu())
    model.add_layer(phidnet.CNN.layer.Linear(100, 10))
    model.add_layer(phidnet.CNN.layer.Relu())
    model.add_layer(phidnet.CNN.layer.Output())

    model.set_optimizer(phidnet.CNN.optimizer.Adam(0.0001))
    return model



model = create_linear_model()
model.fit(X, T, epochs=10, plot=True, batch_size=1000, print_rate=1, val_x=X_test, val_y=T_test, val_size=20)



number = 111
predicted = model.predict(X_test[number])
print('predict:', predicted)
print('predict:', np.argmax(predicted))
print("right answer:", T_test[number])
img = X_test[number].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()
###############################################