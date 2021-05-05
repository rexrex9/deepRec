from utils import osUtils as ou
import random
from tqdm import tqdm
from data_set import filepaths as fp
import pandas as pd

def readRecData(path,test_ratio = 0.1):
    df = pd.read_csv(path,sep='\t',header=None)
    a = df.sort_values(by=[0,3],axis=0)
    a.to_csv('a.csv')
    print(a)
    return

if __name__ == '__main__':
    readRecData(fp.Ml_latest_small.RATING_TS)