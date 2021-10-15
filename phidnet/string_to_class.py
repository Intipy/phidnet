import phidnet.activation


def convert(str):
    if str == 'Sigmoid':
        return phidnet.activation.Sigmoid()
    elif str == 'Softmax':
        return phidnet.activation.Softmax()
    elif str == 'Relu':
        return phidnet.activation.Relu()
    elif str == 'Linear':
        return phidnet.activation.Linear()