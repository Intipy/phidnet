import numpy as np
from matplotlib import pyplot as plt
import phidnet



_, _, X_test, T_test = phidnet.datasets.mnist.load()
X_test = X_test / 255
T_test = phidnet.one_hot_encode.encode_array(T_test, length=10)



phidnet.load.model('E:\Programming\Project\phidnet\examples\ANN_saved_model.pickle')



y = phidnet.model.predict(X_test)
error = phidnet.error.mean_squared_error(y, T_test) / T_test.shape[0]
acc = phidnet.model.accuracy(y, T_test)
print("erorr(trained model):", error)
print("accuracy(trained model):", acc)



for number in range(100, 105):
    predicted = phidnet.model.predict(X_test[number], exponential=False, precision=2)   # Default: exponential=True, precision=6
    print('predict:', predicted)
    print('predict:', np.argmax(predicted))
    print("right answer:", T_test[number])
    img = X_test[number].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.show()
    print("============================")
