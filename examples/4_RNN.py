import numpy as np; np.random.seed(121);
from matplotlib import pyplot as plt

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
print("============================")


