# decision tree

def import_data(name) :
    # import data from source file and store into dataSet
    # input : name - name of file
    # output : trainData - attributes set of each examples
    #          trainLabel - label of each examples
    
    fo = open(name)
    lenses = [listr.strip().split('\t') for listr in fo.readlines()]
    trainData = [x[0:-1] for x in lenses]
    trainLabel = [x[-1] for x in lenses]

    return  [trainData, trainLabel]
# --- end of import_data ---

def entropy(x) :
    # calculate entropy of list x
    # input : x - column vector
    # output : etp - value of entropy

    etp = 0 
    N = len(x)
    for u in set(x) :
        tmp = x.count(u)
        p = 1.0*tmp/N
        etp += -1*p*log(p,2)

    return etp
# --- end of entropy ---

def information_gain(attributes, label) :
    # calculate information gain of attributes with label
    # input : attributes - attributes of each examples
    #         label - label of each examples
    # output : infogain - value of information gain

    Etp = entropy(label)
    condiEtp = 0
    N = len(attributes)
    for u in set(attributes) :
        n = attributes.count(u)
        x = [] 
        for i in range(len(attributes)) :
            if attributes[i] == u :
                x.append(label[i])
        condiEtp += 1.0*n/N*entropy(x) 

    return Etp-condiEtp
# --- end of information_gain ---

def information_gain_rate(attributes, label) :
    # calculate information gain of attributes with label
    # input : attributes - attributes of each examples
    #         label - label of each examples
    # output : infogain - value of information gain

    info_gain = information_gain(attributes, label)
    attr_etp = entropy(attributes)

    return 1.0*info_gain/attr_etp
# --- end of information_gain_rate ---

def split_data(dataSet, label, attr_name, maxAttr) :
    # split data with max attributes
    # input : dataSet - attributes set of examples
    #         label - label of examples
    #         maxAttr - max attributes
    # output : splitedSet - splited dataset
    #          splitedLabel - splited label
    #          sub_attr_name - remove maxAttr then remain attr_name

    choosed_attr = range(len(dataSet[0]))
    del choosed_attr[maxAttr]
    splitedSet = []
    splitedLabel = []
    sub_attr_name = []
    for i in range(len(choosed_attr)) :
        sub_attr_name.append(attr_name[choosed_attr[i]])
    for u in set([tmp[maxAttr] for tmp in dataSet]) :
        tmpSet = []
        subSet = []
        subLabel = []
        for i in range(len(dataSet)) :
            if dataSet[i][maxAttr] == u :
                tmpSet.append(dataSet[i])
                subLabel.append(label[i])
        for i in range(len(tmpSet)) :
            t = [] ;
            for j in range(len(choosed_attr)) :
                t.append(tmpSet[i][choosed_attr[j]])
            subSet.append(t)
        splitedSet.append(subSet)
        splitedLabel.append(subLabel)
    
    return [splitedSet, splitedLabel, sub_attr_name]
# --- end of split_data ---

def constr_ID3_tree(dataSet, label, attr_name) :
    # construct ID3 desision tree
    # input : dataSet - attributes set of examples
    #         label - label of examples
    # output : tree - subtree with dataSet and label

    if len(set(label)) == 1 :
        return label[0]
    if len(dataSet[0]) == 0 :
        number = [[mem, label.count(mem)] for mem in set(label)]
        t = [tmp[-1] for tmp in number]
        return number[t.index(max(t))][0]
    info_gain = [] ;
    for i in range(len(dataSet[0])) :
        info_gain.append(information_gain([tmp[i] for tmp in dataSet], label))
    maxAttr = info_gain.index(max(info_gain))
    [splitedSet, splitedLabel, sub_attr_name] = split_data(dataSet, label, attr_name, maxAttr)
    value_set = set([tmp[maxAttr] for tmp in dataSet])
    i = 0
    tree = {}
    for u in value_set :
        tree[u] = constr_ID3_tree(splitedSet[i], splitedLabel[i], sub_attr_name)
        i += 1
    return {attr_name[maxAttr] : tree}
# --- end of constr_ID3_tree(recursion) ---

def constr_C45_tree(dataSet, label, attr_name) :
    # construct ID3 desision tree
    # input : dataSet - attributes set of examples
    #         label - label of examples
    # output : tree - subtree with dataSet and label

    if len(set(label)) == 1 :
        return label[0]
    if len(dataSet[0]) == 0 :
        number = [[mem, label.count(mem)] for mem in set(label)]
        t = [tmp[-1] for tmp in number]
        return number[t.index(max(t))][0]
    info_gain = [] ;
    for i in range(len(dataSet[0])) :
        info_gain.append(information_gain_rate([tmp[i] for tmp in dataSet], label))
    maxAttr = info_gain.index(max(info_gain))
    [splitedSet, splitedLabel, sub_attr_name] = split_data(dataSet, label, attr_name, maxAttr)
    value_set = set([tmp[maxAttr] for tmp in dataSet])
    i = 0
    tree = {}
    for u in value_set :
        tree[u] = constr_ID3_tree(splitedSet[i], splitedLabel[i], sub_attr_name)
        i += 1
    return {attr_name[maxAttr] : tree}
# --- end of constr_C45_tree(recursion) ---

from math import log
[trainData, trainLabel] = import_data('lenses.txt')
attr_name = ['age', 'prescript', 'astigmatic', 'tearRate']
ID3_tree = constr_ID3_tree(trainData, trainLabel, attr_name)
C45_tree = constr_C45_tree(trainData, trainLabel, attr_name)
print 'ID3 tree : '
print ID3_tree
print '\nC45 tree : '
print C45_tree
infomation_gain([1,1,1,1,2,2,2,2],[1,1,1,1,2,2,2,2])
