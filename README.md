# Phidnet

---------

## 1. Introduction to phidnet
  * Phidnet is a library developed for neural network construction for deep learning, machine learning, and statistics.

---------

## 2. Install phidnet
  * `pip install phidnet`
  * PyPI: https://pypi.org/project/phidnet/
  * GitHub: https://github.com/Intipy/phidnet

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
    + Will be replaced by the built-in matrix library of the phidnet.

  * Configuration of the Phidnet
    + phidnet.activation
    + phidnet.optimizer
    + phidnet.load
    + phidnet.matrix
    + phidnet.set
    + phidnet.one_hot_encode
    + phidnet.model

  * Define optimizer
    + SGD = phidnet.optimizer.SGD(lr=0.01)  # lr: learning rate
    + Momentum = phidnet.optimizer.Momentum(lr=0.01, momentum=0.9)
    + AdaGrad = phidnet.optimizer.AdaGrad(lr=0.01)

  * Set layer
    + phidnet.set.layer(784)
    + phidnet.set.layer(200, activation=phidnet.activation.relu())
    + phidnet.set.layer(10, activation=phidnet.activation.Softmax())
    + If you did not set the activation function, that layer becomes input layer(Input layer does not have activation function.) and if you want to build hidden & output layer, you need to set activation function.
    + In this case, the structure is as follows.
    + 784(input) - 200 - relu - 10 - softmax(output)

  * Compile neural network 
    + phidnet.set.compile(input=X, target=T)
    + If you built the model, you can compile that model with setting input and output data.
  
  * Set test dataset
    + phidnet.set.test(input=X_test, target=T_test)
    + If you want to calculate loss of test dataset(val_loss=True), you need to set this.

  * Fit model
    + phidnet.model.fit(epoch=30, optimizer=SGD, batch=5000, val_loss=True, print_rate=2) 
    + In the example, train the model for epoch. 
    + SGD is the instance of phidnet.optimizer.SGD() class. 
    + Batch size is 5000. 
    + val_loss is loss of test dataset. This helps prevent overfitting. but, calculating this makes the fitting slow.
    + Every 2 epoch, print the loss and accuracy of model(print rate).

  * Predict
    + predicted = phidnet.model.predict(input, exponential=True, precision=2)
    + In the example, the model returns the predicted value in the predicted variable. If exponential= is True, the model returns exponential representation value like 1e-6. When exponential=False, The model returns the value represented by the decimal like 0.018193. The model returns precise values as set to precision. When output is 0.27177211, precision=3, output is 0.271.

  * Save
    + You can save the model with .pickle file.
    + phidnet.save.model('saved_model')
    + It saves trained model in current directory.
    + phidnet.save.model('saved_model', dir='C:\examples')
    + It saves trained model in C:\examples directory.

  * Load
    + phidnet.load.model('C:\examples\saved_model.pickle')
    + You can load trained model.

  * View fitting
    + phidnet.model.show_loss()
    + It shows a change in loss and validation loss.
    + phidnet.model.show_accuracy()
    + It shows a change in accuracy.

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
    + mat + 1, mat * 2, mat / 3

  * Dot product
    + phidnet.matrix.dot(mat1, mat2)
  
  * Index of matrix
    + If you used a regular Python index, it is not suitable for two-dimensional arrays. 
 For example, [1][2] does not point to row 1 and column 2. A two-dimensional array is a shape with an array in array, and Python views the array as one element.
    + The solution is to use the indexing, slicing functions built into the phidnet.
    + Python index: mat[1][2] (does not point to row 1, column 2)
    + Phidnet index: mat("1,2") (point to row 1, column 2)
    + Python slicing: mat[1:3][:8] (does not point row 1-2, column 0-7)
    + Phidnet slicing: mat["1:3,:8"] (point row 1-2, column 0-7)

  * Slicing of matrix(by index)
    + sliced_matrix = phidnet.matrix.slice_full(mat, row_start, row_end, column_start, column_end)
    + sliced_matrix = phidnet.matrix.slice_full(mat, 1, 2, 1, 1)
    + 1-2 row, 1-1 column (0 based index)

  * Slicing of matrix(by python slicing)
    + sliced_matrix = mat["slicing with string"]
    + sliced_matrix = mat["1:3,1:2"]
    + 1-2 row, 1-1 column (0 based index)
    + sliced_matrix = mat[",1:2"]
    + all row, 1-1 column (0 based index)

  * Transpose matrix
    + transposed_matrix = phidnet.Matrix.trans(mat)
    + transposed_matrix = mat.trans()

  * Map
    + def function(x): return 2*x
    + mapped_matrix = phidnet.Matrix.map(mat, function)
  
  * Power
    + pow_matrix = mat ** n
    + mat^n (for every element in matrix)

  * Else
    + .
    + .

---------

## 6. Use phidnet convolution neural network
  * Set layer
    + . 
    + . 
  * writing
    + . 
    + . 

---------

## 7. Use phidnet recurrent neural network
  * Set layer
    + . 
    + . 
  * writing
    + . 
    + . 
    

---------

## 8. Example phidnet
  * Refer to examples for details.
