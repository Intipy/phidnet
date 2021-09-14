from phidnet import network_data



def reset():
    network_data.Loss_list = []  # Record the change in loss
    network_data.Validation_loss_list = []  # Record the change in validation loss
    network_data.Epoch_list = []  # Record the change in Epoch
    network_data.Acc_list = []  # Record the change in accuracy

    network_data.params = {}  # Save weight and bias to save model
    network_data.weight = {}  # Save data of neural network weight in dictionary
    network_data.bias = {}  # Save data of neural network bias in dictionary
    network_data.deltaWeight = {}  # Save data of neural network delta weight in dictionary
    network_data.deltaBias = {}  # Save data of neural network delta bias in dictionary

    network_data.a = {}  # Save data of neural network a in dictionary
    network_data.z = {}  # Save data of neural network z in dictionary

    network_data.X = None  # Save X of neural network
    network_data.target = None  # Save target of neural network
    network_data.X_test = None
    network_data.T_test = None

    network_data.layerNumber = None  # Save number of layer
    network_data.active = []  # Make list that save activation functions of each layers
    network_data.loss = {}  # Save loss of layer

    network_data.layer_shape = []  # Save shape of layer