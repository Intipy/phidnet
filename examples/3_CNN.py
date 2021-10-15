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


# (i, j, k) dimension => j×k size image with i channels
# (n, i, j, k) dimension => j×k size image with i channels (n data)

# (i, j, j) size image, (s, s) size filter n   =========>   (n, j-s+1, j-s+1)
# The number of filters becomes the number of channels after convolution operation
# The image is reduced by adding 1 after subtracting the filter size from the image size (when the stride is 1)
###############################################
phidnet.set.kernel(kernel_number=30, input_channel=1, kernel_size=(5, 5), stride=1, pad=0)
phidnet.set.activation('Relu')
phidnet.set.pooling(size=(2, 2), stride=2)

phidnet.set.layer(4320)
phidnet.set.layer(100, activation='Relu')
phidnet.set.layer(10, activation='Softmax')
phidnet.set.compile(input=X, target=T)
phidnet.set.test(input=X_test, target=T_test)   # If you want to get loss of test data, set the test data and fit the model with "val_loss=True"
###############################################



############################################### Fit model
phidnet.model.fit(epoch=10, optimizer='AdaGrad', lr=0.01, batch=5000, val_loss=False, print_rate=1)   # Showing validation loss make fitting slow
#phidnet.save.model("saved_model")
phidnet.model.show_loss()
phidnet.model.show_accuracy()
###############################################



###############################################
for number in range(100, 200):
    predicted = phidnet.model.predict(X_test[number].reshape(1, 1, 28, 28), exponential=False, precision=2)   # Default: exponential=True, precision=6
    print('predict:', predicted)
    print('predict:', np.argmax(predicted))
    print("right answer:", T_test[number])
    img = X_test[number].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.show()
    print("============================")
###############################################



