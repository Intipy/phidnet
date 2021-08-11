# Phidnet

---------

## 1. Introduction to phidnet
  * Phidnet is a library developed for neural network construction and deep learning.

---------

## 2. Install phidnet
  * `pip install phidnet`
  * PyPI url: https://pypi.org/project/phidnet/

---------

## 3. Requirements of phidnet
  * numpy
  * matplotlib
  * pandas(Optional)

---------

## 4. Use phidnet
  * Import phidnet
    + import phidnet
  * Numpy
    + All data, such as matrix and vector, must be converted to numpy array object.
  * Configuration of the Piednet
    + phidnet.activation
    + phidnet.optimizer
    + phidnet.load
    + phidnet.matrix
    + phidnet.set
    + phidnet.one_hot_encode
    + phidnet.model
  * Define activation function 
    + Sigmoid = phidnet.activation.Sigmoid()
    + Relu = phidnet.activation.Relu()
    + ect
  * Define optimizer
    + SGD = phidnet.optimizer.SGD(lr=0.01)  # lr: learning rate
    + Momentum = phidnet.optimizer.Momentum(lr=0.01, momentum=0.9)
    + AdaGrad = phidnet.optimizer.AdaGrad(lr=0.01)
  * Set data
    + Set input data
      + phidnet.set.input_data(X)
    + Set output data
      + phidnet.set.target_data(T)
  * Set weight and bias
    + phidnet.set.weight(row, column, layer=layer)
    + phidnet.set.bias(column, layer=layer)
    + phidnet.set.weight(2, 10, layer=1)  # 2×10 matrix, 1st layer
    + phidnet.set.bias(10, layer=1)  # 1×10 matrix, 1st layer
  * Build neural network 
    + phidnet.set.build_network(layer)
    + The number of layers is the total number of layers, excluding the input layer. For example, a network with one input layer, one output layer, and one hidden layer in between is a two-layer.
  * Set activation function of neural network 
    + phidnet.set.activation_function(function_list)
    + phidnet.set.activation_function([Sigmoid, Sigmoid])  # 1st layer: Sigmoid, 2nd layer: Sigmoid
    + The example is the activation functions of the two-layer and Sigmoid, an element of list, is the instance of phid.activation.Sigmoid() class
  * Fit model
    + phidnet.model.fit(epoch=1000, optimizer=SGD, print_rate=100, save=True) 
    + In the example, train the model for epoch. SGD is the instance of phid.optimizer.SGD() class. Every 100 epoch, print the loss, accuracy of model(print rate). If save= is true, save weight and bias in pickle. Default: save=False
  * Predict
    + predicted = phidnet.model.predict(input, exponential=True, precision=2)
    + In the example, the model returns the predicted value in the predicted variable. If exponential= is True, the model returns exponential representation value like 1e-6. When exponential=False, The model returns the value represented by the decimal like 0.018193. The model returns precise values as set to precision. When output is 0.27177211, precision=3, output is 0.271.
  * Load
    + phidnet.load.model('C:\examples')
    + If you set it to save=True and trained the model, there would be a file called saved_weight, saved_bias. If the file is in C:\examples\saved_... , you can load trained weight and bias as in the example.
  * View fitting
    + phid.model.show_fit()
    + It shows a change in loss and accuracy.
  * One hot encoding 
    + phidnet.one_hot_encode.encode(number, length=length)
    + phidnet.one_hot_encode.encode(3, length=5)   # [0, 0, 0, 1, 0]
    + phidnet.one_hot_encode.encode_array(array, length=length)
    + phidnet.one_hot_encode.encode_array([[1], [2], [3]], length=5)   # [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
    + phidnet.one_hot_encode.get_number(one_hot_encoded)
    + phidnet.one_hot_encode.get_number([0, 0, 1, 0, 0])   # 2
  * Matrix operations 
    + m = phid.matrix.matrix(list)  # It converts the list into a matrix (※ phidnet matrix object. not numpy object)
    + ect.

---------

## 5. Use phidnet's convolution neural network
  * ect.

---------

## 6. Example phidnet
  * Refer to examples for details.
