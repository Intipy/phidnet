Loss_list = []     # Record the change in loss
Epoch_list = []    # Record the change in Epoch
Acc_list = []      # Record the change in accuracy

weight = {}     # Save data of neural network weight in dictionary
bias = {}     # Save data of neural network bias in dictionary
deltaWeight = {}     # Save data of neural network delta weight in dictionary
deltaBias = {}     # Save data of neural network delta bias in dictionary

a = {}     # Save data of neural network a in dictionary
z = {}     # Save data of neural network z in dictionary

X = None   # Save X of neural network
target = None   # Save target of neural network

layerNumber = None   # Save number of layer
active = []   # Make list that save activation functions of each layers
loss = {}   # Save loss of layer


layer_shape = []   # Save shape of layer