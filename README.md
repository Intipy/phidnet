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

  * Set layer
    + phidnet.set.layer(784)
    + phidnet.set.layer(200, activation=Sigmoid)
    + phidnet.set.layer(10, activation=Sigmoid)
    + If you did not set the activation function, that layer becomes input layer(Input layer does not have activation function.) and if you want to build hidden & output layer, you need to set activation function.

  * Compile neural network 
    + phidnet.set.compile(input=X, target=T)
    + If you built the model, you can compile that model with setting input and output data.

  * Fit model
    + phidnet.model.fit(epoch=1000, optimizer=SGD, batch=500, print_rate=100, save=True) 
    + In the example, train the model for epoch. SGD is the instance of phid.optimizer.SGD() class. Batch size is 500. Every 100 epoch, print the loss and accuracy of model(print rate). If save= is true, save weight and bias in pickle. Default: save=False

  * Predict
    + predicted = phidnet.model.predict(input, exponential=True, precision=2)
    + In the example, the model returns the predicted value in the predicted variable. If exponential= is True, the model returns exponential representation value like 1e-6. When exponential=False, The model returns the value represented by the decimal like 0.018193. The model returns precise values as set to precision. When output is 0.27177211, precision=3, output is 0.271.

  * Load
    + phidnet.load.model('C:\examples')
    + If you set it to save=True and trained the model, there would be a file called saved_weight, saved_bias. If the file is in C:\examples\saved_... , you can load trained weight and bias as in the example.

  * View fitting
    + phidnet.model.show_fit()
    + It shows a change in loss and accuracy.

  * One hot encoding 
    + phidnet.one_hot_encode.encode(number, length=length)
    + phidnet.one_hot_encode.encode(3, length=5)   # [0, 0, 0, 1, 0]
    + phidnet.one_hot_encode.encode_array(array, length=length)
    + phidnet.one_hot_encode.encode_array([[1], [2], [3]], length=5)   # [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
    + phidnet.one_hot_encode.get_number(one_hot_encoded)
    + phidnet.one_hot_encode.get_number([0, 0, 1, 0, 0])   # 2
    
  * Pre-prepared datasets
    + X, T, X_test, T_test = phidnet.datasets.mnist.load()
    + It loads mnist dataset with 1d shape. (784)
    + X, T, X_test, T_test = phidnet.datasets.mnist.load_2d()
    + It loads mnist dataset with 2d shape. (28, 28)

---------

## 5. Use phidnet matrix
  * Converting to matrix
    + mat = phidnet.array(list)

  * Add, Multiplication, Subtraction
    + Equal to other classes of operations
    + mat1 + mat2, mat1 - mat2, mat1 * mat2

  * Dot product
    + phidnet.matrix.dot(mat1, mat2)

  * Slicing of matrix(by index)
    + sliced_matrix = phidnet.matrix.slice_full(mat, row_start, row_end, column_start, column_end)
    + sliced_matrix = phidnet.matrix.slice_full(mat, 1, 2, 1, 1)
    + 1~2 row, 1~1 column (0 based index)

  * Slicing of matrix(by python slicing)
    + sliced_matrix = mat[ " Python Slicing Grammar " ]
    + sliced_matrix = mat["1:3,1:2"]
    + 1~2 row, 1~1 column (0 based index)
    + sliced_matrix = mat[",1:2"]
    + all row, 1~1 column (0 based index)

  * Transpose matrix
    + transposed_matrix = phidnet.Matrix.trans(mat)
    + transposed_matrix = mat.trans()

  * Map
    + 

---------

## 6. Use phidnet convolution neural network
  * Set layer
    + 
    + 
  * writing
    +
    +

---------

## 7. Example phidnet
  * Refer to examples for details.
