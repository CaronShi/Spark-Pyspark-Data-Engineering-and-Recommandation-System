'''
Method Description
Based on Assignment3 task 2_3, I worked on implementing a collaborative-filtering recommendation system and a xgboost Model-based recommendation system. In the assignment3, I got my rmse around 0.983 and tried to minimize the error by adjusting the xgboost parameters, including learning rate(from 0.05 to 0.02, n_estimators(300 to 1000) and max_depth(from 4 to 6)). However, I found it was hard to hit below 0.98 and the time exceeded the limit as well, so I used other json files, and combined the features together for the model training data, and finally got below 0.98. 

Error distribution
>=0 and <1: 129200
>=1 and <2: 12174
>=2 and <3: 653
>=3 and <4: 17
>=4:0

RMSE:
0.9775617183349736

Execution Time:
1198.77s or 20 mins
'''
import sys
from pyspark import SparkContext
import time
import json
import xgboost as xgb
import random
from datetime import date
from datetime import datetime
from itertools import combinations

sc = SparkContext(appName = "assignment7", master = "local[*]")
output_f =sys.argv[3]#'task2_3_output.csv' #
folder_path =sys.argv[1] #'../resource/asnlib/publicdata' #
test_f =sys.argv[2] #folder_path+'/yelp_val.csv'
train_f = folder_path+'/yelp_train.csv'

sc.setLogLevel("ERROR")

s_t = time.time()


def myLambda (x):
    x_split = x.split (',')
    return  x_split[:2]

rdd_ = sc.textFile(train_f)
header = rdd_.take(1)
rdd_ = rdd_.filter(lambda x: x not in header) #business_id,user_id

def split_data(data):
    data = data.split(',')
    return data[0],data[1],float(data[2])

train_rdd = rdd_.map(split_data)#users_id, business_id,star

user_id = train_rdd.map(lambda x: x[0]).distinct().collect() #['user_id1','qwrewfr'...]

test_rdd = sc.textFile(test_f)
test_header = test_rdd.take(1)
testrdd_ = test_rdd.filter(lambda x: x not in header) #business_id,user_id
test = testrdd_.map(myLambda)#users_id, business_id,star
########CF BASED##########
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
bid_train_map = bid_train.collectAsMap()#{('3MntE_HWbNNoyiLGxywjYA', {'T13IBpJITI32a1k41rc-tg': 5.0})]
selected_col2 = train_rdd.map(lambda x: (x[1], x[2]))
bus_avg = selected_col2.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
        .mapValues(lambda x: x[0] / x[1]).collectAsMap()#{'3MntE_HWbNNoyiLGxywjYA': {'T13IBpJITI32a1k41rc-tg': 5.0},..}

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
    ) [: 111]
    
    score = 0
    if len(candidate_list) > 0:
        weights = [abs(sim) for sim, b2 in candidate_list]
        score = sum([busid_useri[b2][user_id] * sim for sim, b2 in candidate_list]) / sum(weights)
    return score

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

b_sim_step1 = bid_pair.flatMap(p_similarity).groupByKey() 
b_sim_step2 = b_sim_step1.mapValues(dict).sortByKey()
b_sim = b_sim_step2.collectAsMap()


test_cal = test.map(lambda x:  fill(x[0], x[1])).collect()
b1_b2 = test.map(lambda x: (x[0], x[1])).collect()

cf_dict_ = {}
l = []
for i in b1_b2:
    ll = i[0]+','+i[1]
    l.append(str(ll))
cf_dict = dict(zip(l, test_cal))  
#tx = cf_dict['wf1GqnKQuvH-V3QN80UOOQ,fThrN4tfupIGetkrz18JOg']
#print(tx)
#exit()
#print(cf_dict) #'g0LQ3Mzp5jup5w1_UcSKeA,SkSkWld_ijmWVEDo9_ztKw': 4.15
 
print('1st model finished')

##########MODEL BASED##########
##########MODEL BASED##########


def json_l(data):
    return json.loads(data)

def get_rdd(raw_rdd,id_str):
    rdd = raw_rdd.map(json_l).map(lambda x: (x[id_str], 1)).countByKey()
    return rdd

def non_val(data):
  if data: return len(data)
  else: return 0

def data_format(x):
    date1 = datetime.strptime(x, '%Y-%m-%d').date()
    date2 = date(2022, 12, 4)
    d= date2 - date1
    return (d.days)
    
def time_val(data):
    time_v = sum(data['time'].values())
    return data['business_id'], time_v 
train_rdd = sc.textFile(folder_path+"/yelp_train.csv").map(lambda line: line.split(",")).filter(lambda x: x[0]!='user_id').map(lambda x: (x[0], x[1], float(x[2]))).cache()
#train_rdd = rdd_.map(lambda x: (x[0], x[1], float(x[2]))).cache()

user_sides = ["/tip.json","/photo.json", "/checkin.json"]
user_tip = get_rdd(sc.textFile(folder_path+user_sides[0]),'user_id')
bus_tip = get_rdd(sc.textFile(folder_path+user_sides[0]),'business_id')
bus_photo = get_rdd(sc.textFile(folder_path+user_sides[1]),'business_id')

bus_checkin = sc.textFile(folder_path+user_sides[2]).map(json_l).map(time_val).reduceByKey(lambda x,y: x+y).collectAsMap()

def get_user_data(data):
    v = ['user_id',  'review_count','average_stars','funny', 'cool', 'fans', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos'] 
    user_l =  [data[v[1]], data[v[2]], data[v[3]], data[v[4]], data[v[5]], data[v[6]], data[v[7]], data[v[8]], data[v[9]], data[v[10]], data[v[11]], data[v[12]], data[v[13]], data[v[14]], data[v[15]], data[v[16]],non_val(data['friends']), data_format(data['yelping_since']), user_tip[v[1]]]
    return data[v[0]],user_l
    
user_rdd = sc.textFile(folder_path+"/user.json").map(json_l)
user_data = user_rdd.map(get_user_data)
user_dict = user_data.collectAsMap()


# print(user_data.take(5))  
def check_dic(id,dic):
    if id in dic.keys():
        return dic[id]
    else:
        return 0
def get_bus_data(data):
    v = ['business_id', 'review_count','stars','is_open','latitude','longitude','attributes','hours']
    b_l = [data[v[1]], data[v[2]], data[v[3]], data[v[4]], data[v[5]], non_val(data[v[7]]), non_val(data[v[6]]), check_dic(data[v[0]],bus_tip),check_dic(data[v[0]],bus_checkin),check_dic(data[v[0]],bus_photo)]
          
    return  data[v[0]],b_l
bus_rdd = sc.textFile(folder_path+"/business.json").map(json_l)
tem_bus = bus_rdd.map(get_bus_data)  
bus_dict = tem_bus.collectAsMap()


train_data = train_rdd.map(lambda x: (x[0], x[1])).collect()
train_x = list(user_dict[x[0]]+bus_dict[x[1]] for x in train_data)

train_y = train_rdd.map(lambda x: x[2]).collect()
test = sc.textFile(test_f).map(lambda line: line.split(",")).filter(lambda x: x[0]!='user_id')
test_rdd = test.map(lambda x: (x[0], x[1])).cache()
test_data = test_rdd.collect()
print('testdata')
test_x = [user_dict[x[0]]+bus_dict[x[1]] for x in test_data]

val_use = test.map(lambda x: ((x[0], x[1]),x[2])).collectAsMap()

##########model#############
model = xgb.XGBRegressor(n_estimators = 800, max_depth =6, eval_metric = 'rmse', learning_rate = 0.05, random_state= 100)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)

print('2nd model finished')

###CF:test_cal, MODEL: y_pred
def final_score(ids, y_pred):

    neigbor_dict = dict()
    for i in ids:#cal how many users review this business
        bid = i[0]
        uid = i[1]
        if bid not in neigbor_dict:
            neigbor_dict[bid] = set([uid])
        else: 
            neigbor_dict[bid].add(uid)
  
    ids_pre = dict()
    count_level1 = 0
    count_level2 = 0
    count_level3 = 0
    
    count = 0
    for i in range(0,len(ids)):
        bid = ids[i][0]
        uid = ids[i][1]
        if len(neigbor_dict[bid]) > 100: #more weight give to model if this business have 100 more reviews
            model_weight = 0.99
            count_level1+=1
        elif len(neigbor_dict[bid]) > 20:
            model_weight =0.98
            count_level2+=1
        else:
            model_weight =0.95
            count_level3+=1
        cf_pred = cf_dict[str(uid)+','+str(bid)]
        
        final_weight = (cf_pred* (1 - model_weight)) + (y_pred[i] *model_weight)
   
        ids_pre[uid+','+bid] = final_weight
    print(' count_123:', count_level1, count_level2, count_level3)
    return ids_pre #{ids: pre}

ids = []
for i in range(len(y_pred)):
    user_id = test_data[i][0]
    bus_id = test_data[i][1]
    id_ = [str(bus_id), str(user_id)]
    ids.append(id_)
print('ids', ids[:2])

ids_pre = final_score(ids,y_pred)
f_id_keys = list(ids_pre.keys())
f_id_val = list(ids_pre.values())

with open(output_f, 'w') as f:
    f.write("user_id, business_id, prediction\n")
    for i in range(len(y_pred)):
        f.write(str(f_id_keys[i])+','+ str(f_id_val[i]) + '\n')

y_pred = f_id_val
dif = 0
n = len(y_pred)

for i in range(len(y_pred)): 
    key_val = (test_data[i][0], test_data[i][1])
    dif += (float(y_pred[i])- float(val_use[key_val]))**2
avg = dif/n
rmse = avg**0.5

                          
print('rmse',rmse)
print("Duration: ", float(time.time() - s_t)/60)

