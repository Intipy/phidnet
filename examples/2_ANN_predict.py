import numpy as np
from matplotlib import pyplot as plt
import phidnet



_, _, X_test, T_test = phidnet.datasets.mnist.load()
X_test = X_test / 255



phidnet.set.layer(784)
phidnet.set.layer(200, activation=phidnet.activation.Relu())
phidnet.set.layer(10, activation=phidnet.activation.Softmax())
phidnet.set.compile()
phidnet.load.model('E:\Programming\Project\phidnet\examples\saved_model.pickle')



for number in range(100, 200):
    predicted = phidnet.model.predict(X_test[number], exponential=False, precision=2)   # Default: exponential=True, precision=6
    print('predict:', predicted)
    print('predict:', np.argmax(predicted))
    print("right answer:", T_test[number])
    img = X_test[number].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.show()
    print("============================")
