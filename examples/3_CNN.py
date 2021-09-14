import numpy as np; np.random.seed(121)
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



############################################### Optimizer & Activation function setting
SGD = phidnet.CNN.optimizer.SGD(lr=0.01)
Momentum = phidnet.CNN.optimizer.Momentum(lr=0.01, momentum=0.9)
AdaGrad = phidnet.CNN.optimizer.AdaGrad(lr=0.01)
###############################################



# (i, j, k) dimension => j×k size image with i channels
# (n, i, j, k) dimension => j×k size image with i channels (n data)

# (i, j, j) size image, (s, s) size filter n   =========>   (n, j-s+1, j-s+1)
# The number of filters becomes the number of channels after convolution operation
# The image is reduced by adding 1 after subtracting the filter size from the image size (when the stride is 1)
###############################################
phidnet.CNN.set.filter(filter_number=30, input_channel=1, filter_size=5, stride=1, pad=0)
phidnet.CNN.set.activation(phidnet.activation.Relu())
phidnet.CNN.set.pooling(height=2, width=2, stride=2)

phidnet.CNN.set.layer(4320)
phidnet.CNN.set.layer(100, activation=phidnet.activation.Relu())
phidnet.CNN.set.layer(10, activation=phidnet.activation.Softmax())
phidnet.CNN.set.compile(input=X, target=T)
phidnet.CNN.set.test(input=X_test, target=T_test)   # If you want to get loss of test data, set the test data and fit the model with "val_loss=True"
###############################################



############################################### Fit model
phidnet.CNN.model.fit(epoch=10, optimizer=AdaGrad, batch=5000, val_loss=False, print_rate=1)   # Showing validation loss make fitting slow
#phidnet.save.model("saved_model")
phidnet.CNN.model.show_loss()
phidnet.CNN.model.show_accuracy()
###############################################



###############################################
for number in range(100, 200):
    predicted = phidnet.CNN.model.predict(X_test[number].reshape(1, 1, 28, 28), exponential=False, precision=2)   # Default: exponential=True, precision=6
    print('predict:', predicted)
    print('predict:', np.argmax(predicted))
    print("right answer:", T_test[number])
    img = X_test[number].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.show()
    print("============================")
###############################################
