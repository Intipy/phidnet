import numpy as np; np.random.seed(121)
import phidnet   # Import phidnet



############################################### Data
X, T, X_test, T_test = phidnet.datasets.mnist.load()
X, X_test = X / 255, X_test / 255
T = phidnet.one_hot_encode.encode_array(T, length=10)
T_test = phidnet.one_hot_encode.encode_array(T_test, length=10)
print("input shape:", X.shape)
print("output shape:", T.shape)
print("test input shape:", X_test.shape)
print("test output shape:", T_test.shape)
###############################################



############################################### Build neural network
phidnet.set.layer(784)
phidnet.set.layer(200, activation='Relu')
phidnet.set.layer(10, activation='Softmax')
phidnet.set.compile(input=X, target=T)
phidnet.set.test(input=X_test, target=T_test)   # If you want to get loss of test data, set the test data and fit the model with "val_loss=True"
###############################################

print(phidnet.network_data.layer)

############################################### Fit model
phidnet.model.fit(epoch=2, optimizer='AdaGrad', lr=0.01, batch=128, val_loss=True, print_rate=1)   # Showing validation loss make fitting slow
# phidnet.save.model("ANN_saved_model")
phidnet.model.show_loss()
phidnet.model.show_accuracy()
###############################################
