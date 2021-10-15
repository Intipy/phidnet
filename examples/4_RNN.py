import numpy as np; np.random.seed(121);

import phidnet   # Import phidnet



sentence = 'you say goodbye and i say hello.'.replace('.', ' .')
sentence_split = sentence.split(' ')



word_bag = []
for i in sentence_split:
    if i in word_bag:
        pass
    else:
        word_bag.append(i)



sentence_dic = {}
for i in range(len(word_bag)):
    sentence_dic[word_bag[i]] = i



X = []
for i in sentence_split:
    enc = phidnet.one_hot_encode.encode(sentence_dic[i], length=len(word_bag))
    X.append(enc)

X = np.array(X, dtype='int32')

T = np.delete(X, 0, axis=0)
T = np.append(T, np.array([[0, 0, 0, 0, 0, 0, 0]], dtype='int32'), axis=0)

T = T.reshape(1, T.shape[0], T.shape[1])
X = X.reshape(1, X.shape[0], X.shape[1])


encoded_dic = {}
for i in sentence_dic.keys():
    encoded_dic[i] = phidnet.one_hot_encode.encode(sentence_dic[i], length=len(word_bag)).astype('int32')
encoded_dic['None'] = [0, 0, 0, 0, 0, 0, 0]



print(sentence)       # you say goodbye and i say hello .
print(sentence_split) # ['you', 'say', 'goodbye', 'and', 'i', 'say', 'hello', '.']
print(word_bag)       # ['you', 'say', 'goodbye', 'and', 'i', 'hello', '.']
print(sentence_dic)   # {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
print(encoded_dic)
print(X)
print(T)
print(X.shape)
print(T.shape)
print("============================")



vocab_size, wordvec_size, hidden_size = 1, 2, 3   # v, d, h
np.set_printoptions(precision=3, suppress=True)


phidnet.set.rnn(7, 7)   # (???, sentence length)
phidnet.set.compile(input=X, target=T)

epoch = 4
for e in range(epoch):
    forwarded = phidnet.network_data.layer[0].forward(X).reshape(8, 7)
    backwarded = phidnet.network_data.layer[0].backward(forwarded - T)

    for i in range(3):
        phidnet.network_data.layer[0].params[i] -= 0.0001 * phidnet.network_data.layer[0].grads[i]


'''
for i in range(len(forwarded)):
    forwarded[i][np.argmax(forwarded[i])] = 1
for i in range(len(forwarded)):
    for j in range(len(forwarded[0])):
        if forwarded[i][j] != 1:
            forwarded[i][j] = 0
'''

print(forwarded)



'''
###############################################
phidnet.set.rnn(7, 10)
phidnet.set.nn(1, 10)
phidnet.set.compile(input=X, target=T)
###############################################



############################################### Fit model
phidnet.model.fit(epoch=10, optimizer=AdaGrad, batch=1, val_loss=False, print_rate=1)
#phidnet.save.model("saved_model")
phidnet.model.show_loss()
phidnet.model.show_accuracy()
###############################################
'''
