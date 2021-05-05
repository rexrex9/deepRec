from utils import osUtils as ou
import random
from tqdm import tqdm

def readRecData(path,test_ratio = 0.1):
    print('读取用户评分三元组...')
    user_set,item_set=set(),set()
    triples=[]
    for u, i, r in tqdm(ou.readTriple(path)):
        user_set.add(int(u))
        item_set.add(int(i))
        triples.append((int(u),int(i),int(r)))

    test_set=random.sample(triples,int(len(triples)*test_ratio))
    train_set=list(set(triples)-set(test_set))

    #返回用户集合列表，物品集合列表，与用户，物品，评分三元组列表
    return list(user_set),list(item_set),train_set,test_set

def testSetForTopKevaluation(testSet):
    all_testItems = set()
    user_items = dict()
    for u,v,r in testSet:
        all_testItems.add(v)
        if u not in user_items:
            user_items[u]={
                'pos':set(),
                'neg':set()
            }
        if r==1:
            user_items[u]['pos'].add(v)
        else:
            user_items[u]['neg'].add(v)
    return all_testItems, user_items

if __name__ == '__main__':
    pass