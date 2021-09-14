from phidnet.CNN import network_data



def reset():
    network_data.layer = []
    network_data.affine_shape = []
    network_data.active = []
    network_data.layer_weight_index = []

    network_data.X = None
    network_data.target = None
    network_data.X_test = None
    network_data.T_test = None

    network_data.Loss_list = []
    network_data.Validation_loss_list = []
    network_data.Epoch_list = []
    network_data.Acc_list = []