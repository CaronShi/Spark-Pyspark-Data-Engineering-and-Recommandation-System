#Task1 Jaccard based LSH 
#1. contain all business pairs
#2. The header is business_id_1, business_id_2, similarity
#3. pairs in lexicographical order sorted

import os,sys,time
from pyspark import SparkContext,SparkConf
from itertools import combinations
import random

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
start = time.time()
conf = SparkConf().setAppName('task1').setMaster('local[*]')
sc = SparkContext(conf=conf)
output_f ='output.csv'
#load file and preprocess
path = 'sample_train.csv' #sys.argv[] 
#output_path = 'C:/Users/13412/Downloads/553HW/task1_output.csv' #sys.argv[] 

s_time = time.time()
rdd_ = sc.textFile(path).map(lambda x: (x.split(',')[0], x.split(',')[1])) #takes out only users_id, business_id

#header = rdd_.take(1)
#rdd = rdd_.filter(lambda x: x not in header).cache()
users = rdd_.map(lambda x: x[0]).map(lambda x: (x,1)).groupByKey().zipWithIndex()
print(users)
exit()
#userNum = users.count()
characters = rdd.leftOuterJoin(users).map(lambda x: x[1]).groupByKey().mapValues(set)

###characters = rdd.leftOuterJoin(users).map(lambda x: x[1]).reduceByKey(lambda x,y:x+y).mapValues(lambda x: {x})
x_rdd = characters.collectAsMap()
#print(characters.collectAsMap(),'rgrr',characterRDD)
#exit()
M = len(users.collect())

def minHash(data):
    r = 2
    N = 120
    A = [i for i in range(1, N+1)]
    B = [random.randint(10, 25555) for i in range(N)]
    s = []

    for i in range(N):
        for u in data:
            min_val = (A[i]) * u + B[i] % M 
        s.append(min(min_val))

        res = [tuple(s[i: i + r]) for i in range(0, N, r)]
    return res

def sim(x):
    x = sorted(x)
    a, b = x
    rdd = x_rdd
    sa, sb = rdd[a], rdd[b]
    cross = len(sa.intersection(sb))
    union = len(sa.union(sb))
    RES = (a,b, cross / union)
    return RES

if __name__ == '__main__':

    bus_col = characters.mapValues(minHash)
    #[('lQpxpk_ZFJ_ZtYNYhJQv8Q', [(17, 11),  (19, 16), (5, 26), (6, 18)])...]
    threshold = 0.5
    b = 60
    cand_col1 = bus_col.flatMap(lambda x: [(hash((i, *x[1][i])), x[0]) for i in range(b)])
    cand_col = cand_col1.map(lambda x: (x[0], [''.join(list(x[1]))])).reduceByKey(lambda x,y: x+y).filter(lambda x: len(x[1]) > 1)
 
    cand_col2 = cand_col.flatMap(lambda x: list(combinations(x[1], 2))).distinct()
    print(cand_col2.take(5))
    final = cand_col2.map(sim).filter(lambda x: x[2] >= threshold).collect()
    print(final)
    with open(output_f, 'w+') as output:
        output.write("business_id_1,business_id_2,similarity\n")
        res = sorted(final)
        for i in sorted(res):
            output.write(str(i[0])+"," + str(i[1])+"," + str(i[2])+'\n')
    output.close()

    print("duration: ", time.time()- start)