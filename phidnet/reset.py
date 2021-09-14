from phidnet import network_data



def reset():
    network_data.layer = []

    network_data.affine_shape = []
    network_data.active = []
    network_data.layer_weight_index = []

    network_data.X = None  # Save X of neural network
    network_data.target = None  # Save target of neural network
    network_data.X_test = None
    network_data.T_test = None

    network_data.Loss_list = []  # Record the change in loss
    network_data.Validation_loss_list = []  # Record the change in validation loss
    network_data.Epoch_list = []  # Record the change in Epoch
    network_data.Acc_list = []  # Record the change in accuracy