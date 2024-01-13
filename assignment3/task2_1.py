#task 2

import os,sys,time
from pyspark import SparkContext,SparkConf
from itertools import combinations
import random

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
start = time.time()
conf = SparkConf().setAppName('task2.1').setMaster('local[*]')
sc = SparkContext(conf=conf)
output_f ='task2_1_output.csv'
train_path = 'sample_train.txt'  #../resource/asnlib/publicdata/yelp_train.csv
test_path = 'sample_train.txt'  
###similarity task1

def myLambda (x):
    x_split = x.split (',')
    n = len(x_split)
    x1 = x_split[0] if n>0 else 'None'
    x2 = x_split[1] if n>1 else 'None'
    x3 = x_split[2] if n>2 else 0.0
    return (x1, x2, x3)


def x_preprocess(x):
    x_split = x.split (',')
    return  x_split[:2]

#output_path = 'C:/Users/13412/Downloads/553HW/task1_output.csv' #sys.argv[] 
s_time = time.time()
rdd_ = sc.textFile(train_path)
header = rdd_.take(1)
rdd_ = rdd_.filter(lambda x: x not in header) #business_id,user_id
train_rdd = rdd_.map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2])))#users_id, business_id,star

user_id = train_rdd.map(lambda x: x[0]).distinct().collect() #['user_id1','qwrewfr'...]
#users = user_id.zipWithIndex()#[('user_id1', 0), ('-qj9ouN0bzMXz1vfEslG-A', 1)]

test_rdd = sc.textFile(test_path)
test_header = test_rdd.take(1)
testrdd_ = test_rdd.filter(lambda x: x not in header) #business_id,user_id
test = testrdd_.map(x_preprocess)#users_id, business_id,star


def assign_i(data):
    res = []
    for i in range(0, len(user_id)):
        line = (data[i], i)
        res.append(line)
    return res

data = assign_i(user_id)
users = sc.parallelize(data)
characters = train_rdd.leftOuterJoin(users).map(lambda x: x[1]).groupByKey().mapValues(set) # [('9fBwxdHyJRS1nLkMJ8cavw': {131}),...]
busid_useri= characters.collectAsMap()  #x_rdd
# {'business_id': {user_index}, 'GPX3TnZ0-4pAxKPJUESbeA': {16,1}...}


selected_col = train_rdd.map(lambda x:(x[0], x[2]))
users_sum = selected_col.aggregateByKey((0, 0), # initial value (sum, count) 
                           lambda x, y: (x[0] + y, x[1] + 1), # combine values 
                           lambda x, y: (x[0] + y[0], x[1] + y[1])) # combine sums 
users_avg = users_sum.mapValues(lambda x: x[0] / x[1]).collectAsMap()
selected_cols = train_rdd.map(lambda x: (x[1], (x[0], x[2])))
bid_train = selected_cols.map(lambda x: (x[0], [x[1]])) \
    .reduceByKey(lambda x, y: x + y) \
    .mapValues(dict).cache()
bid_train_map = bid_train.collectAsMap()
#{('3MntE_HWbNNoyiLGxywjYA', {'T13IBpJITI32a1k41rc-tg': 5.0}), ('YXohNvMTCmGhFMSQsDZq1g', {'4bQqil4770ey8GfhBgEGuw': 5.0, 'CkqT3yGUeM_9vSbFES_O5w': 5.0})]
selected_col2 = train_rdd.map(lambda x: (x[1], x[2]))
bus_avg = selected_col2.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
        .mapValues(lambda x: x[0] / x[1]).collectAsMap()
#{'3MntE_HWbNNoyiLGxywjYA': {'T13IBpJITI32a1k41rc-tg': 5.0},...}

###############################
M = len(users.collect())
r = 2
N = 111
A,B = [], []
for i in range(1,N+1):
    A.append(i)
for i in range(N):
    B.append(random.randint(100, 20000))


def minHash(data):
    s = []
    for i in range(N):
        min_val= M + 1
        for u in data:
            formula = ((A[i]) * u + B[i]) % M
            if formula < min_val:
                min_val = formula
                s.append(min_val)

        res = [tuple(s[i: i + r]) for i in range(0, N, r)]
    return res


def sim(x):
    x = sorted(x)
    a, b = x
    rdd = busid_useri
    sa, sb = rdd[a], rdd[b]
    cross = len(sa.intersection(sb))
    union = len(sa.union(sb))
    RES = (a,b, cross / union)
    return RES
def p_similarity(pair): 
    bid_1, bid_2 = pair 
    bid1_usersid = bid_train_map[bid_1] 
    bid2_usersid = bid_train_map[bid_2] 
    empty = []
    common_userset = set(bid1_usersid.keys()).intersection(bid2_usersid.keys()) 
    if not common_userset: return set()

    bid_1_avg = sum(bid1_usersid.values()) / len(common_userset)
    bid_2_avg = sum(bid2_usersid.values()) / len(common_userset)
    bid_1_sq = sum([(bid1_usersid[u] - bid_1_avg)**2 for u in common_userset])
    bid_2_sq = sum([(bid2_usersid[u] - bid_2_avg)**2 for u in common_userset])
    bid_1bid_2 = sum([(bid1_usersid[u] - bid_1_avg)*(bid2_usersid[u] - bid_2_avg) for u in common_userset])
    if bid_1bid_2 == 0: return empty
    d = (bid_1_sq * bid_2_sq) ** 0.5
    p_sim = bid_1bid_2 / d

    if p_sim <= 0: return empty
    return [(bid_1, (bid_2, p_sim)), (bid_2, (bid_1, p_sim))]

bus_col = characters.mapValues(minHash)
#[('lQpxpk_ZFJ_ZtYNYhJQv8Q', [(17, 11),  (19, 16), (5, 26), (6, 18)])...]
b = 40
cand_col1 = bus_col.flatMap(lambda x: [(hash((i, *x[1][i])), x[0]) for i in range(b)])
cand_col = cand_col1.map(lambda x: (x[0], [''.join(list(x[1]))])).reduceByKey(lambda x,y: x+y).filter(lambda x: len(x[1]) > 1)
cand_col2 = cand_col.flatMap(lambda x: list(combinations(x[1], 2))).distinct()
final = cand_col2.map(sim).filter(lambda x: x[2] >= 0.5)
bid_pair = final.map(lambda x: (x[0], x[1]))
#[('1zMzkxYgVqJbKGTMQorarA', '56_j_lcGj5X9SpM2KzLm4A'), ('364hhL5st0LV16UcBHRJ3A', 'cYwJA2A6I12KNkm2rtXd5g'),
#print('signle',bid_train_map['1zMzkxYgVqJbKGTMQorarA']) #get user id:stars


#b_sim_step1 = bid_pair.flatMap(p_similarity)
#print('step1',b_sim_step1.collect() )

b_sim_step1 = bid_pair.flatMap(p_similarity).groupByKey() 
b_sim_step2 = b_sim_step1.mapValues(dict).sortByKey()
b_sim = b_sim_step2.collectAsMap()

def fill(user_id, business_id):

    if business_id in bid_train_map and user_id not in busid_useri: return bus_avg[business_id]
    if business_id not in bid_train_map and business_id in bus_avg: return bus_avg[business_id]
    if business_id not in bid_train_map: return users_avg[user_id]

        
    bSim = bid_train_map[business_id]
    candidate_set = set(bSim.keys()).intersection(busid_useri[user_id])
    if len(candidate_set) <= 0: return bus_avg[business_id]

    candidate_list = sorted (
        [ (bSim[candidate], candidate) for candidate in candidate_set ],
        key = lambda x: x[0], 
        reverse=True
    ) [: N]
    
    
    score = 0
    if len(candidate_list) > 0:
        weights = [abs(sim) for sim, b2 in candidate_list]
        score = sum([busid_useri[b2][user_id] * sim for sim, b2 in candidate_list]) / sum(weights)
    return score

test_cal = test.map(lambda x:  fill(x[0], x[1])).collect()
b1_b2 = test.map(lambda x: (x[0], x[1])).collect()
with open(output_f, 'w') as f:
    f.write("user_id, business_id, prediction\n")
    for i in range(0,len(b1_b2)):
        f.write(str(b1_b2[i][0])+','+str(b1_b2[i][1])+','+ str(test_cal[i])+'\n')

val_rdd = sc.textFile(test_path).map(lambda line: line.split(",")).filter(lambda x: x[0]!='user_id').map(lambda x: (x[0], x[1], float(x[2]))).collect()
n = len(test_cal)
temp = 0
for i in range(n):
    p = float(test_cal[i])
    r = float(val_rdd[i][2])
    temp = temp + (p - r) ** 2
print("RMSE", (temp / n) ** .5)

#business_id_1,business_id_2,similarity
