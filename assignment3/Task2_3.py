import sys
from pyspark import SparkContext
import time
import json
import xgboost as xgb
import random
from itertools import combinations

sc = SparkContext(appName = "assignment3 task2_3", master = "local[*]")

output_f =sys.argv[3]#'task2_3_output.csv' #
folder_path =sys.argv[1] #'../resource/asnlib/publicdata' #
test_f =sys.argv[2] #folder_path+'/yelp_val.csv'
train_f = folder_path+'/yelp_train.csv'

sc.setLogLevel("ERROR")

s_time = time.time()
########CF BASED##########
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

#######NOT USEFUL WRITE OUT
with open('task2_3_output_cf_dict.csv', 'w') as f:
    f.write(str(cf_dict)  )
f.close()    
print('1st model finished')
#######NOT USEFUL WRITE OUT END


##########MODEL BASED##########
##########MODEL BASED##########
##########MODEL BASED##########
###input files
def split_data_(data):
    data = data.split(',')
    return data[0], (data[1],float(data[2]))
train_rdd = rdd_.map(split_data_)
#########test###########
def split_testdata(data):
    data = data.split(',')
    return data[0], data[1]
test_rdd = testrdd_.map(split_testdata)
#########features###########
b_rdd = sc.textFile(folder_path + '/business.json').map(lambda x: json.loads(x))
u_rdd = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x))

b_features = b_rdd.map(lambda x: (x['business_id'], (int(x['review_count']), float(x['stars']))))
u_features = u_rdd.map(lambda x: (x['user_id'], (int(x['review_count']), float(x['average_stars']),int(x['useful']))))
max_ureview  = u_features.map(lambda x: x[1][0]).max()
max_breview = b_features.map(lambda x: x[1][0]).max()
max_useful = u_features.map(lambda x: x[1][2]).max()
#min_max scalar(x - min_val) / (max_val - min_val)   
b_features = b_features.map(lambda x: (x[0], (x[1][0]/max_breview, x[1][1]/5)))
u_features = u_features.map(lambda x: (x[0], (x[1][0]/max_ureview, x[1][1]/5, x[1][2]/max_useful)))

def preprocess_uid(x):
    u_id = x[0]
    b_id = x[1][0][0]
    rate = x[1][0][1]
    u_rev = x[1][1][0]
    u_star =  x[1][1][1]
    u_useful =  x[1][1][2]                   
    return b_id,(rate,u_rev,u_star, u_useful)

def preprocess_train(x):#((5.0, 636, 3.97), (69, 3.0 ,3))
    rate = x[0][0]
    b_rev = x[0][1]
    b_star =  x[0][2]
    u_useful = x[0][3]
    u_rev = x[1][0]
    u_star = x[1][1]
    
    return (b_rev,b_star,u_rev,u_star,u_useful),rate


train_step =  train_rdd.leftOuterJoin(u_features).map(preprocess_uid).join(b_features) 
train_process1 = train_step.map(lambda x: x[1]).map(preprocess_train)
train_process =train_process1.collectAsMap()
              
x_train = list(train_process.keys())#(b_rev,b_star,u_rev,u_star,u_useful)
y_rate = list(train_process.values()) #rating'''

#########test###########
def pre_test_uid(x):
    u_id = x[0]
    b_id = x[1][0]
    u_rev = x[1][1][0]
    u_star =  x[1][1][1]
    u_useful =  x[1][1][2]     
    return b_id,(u_id,u_rev,u_star, u_useful)

def process_test(x):
    u_id = x[0]
    b_id = x[1][0][0]
    u_rev = x[1][1][0]
    u_star =  x[1][1][1] 
    
    b_star = x[1][0][2]
    b_rev = x[1][0][1]
    u_useful =  x[1][0][3] 
    return (u_id,b_id),(b_rev, b_star,u_rev, u_star,u_useful)
test_step = test_rdd.join(u_features).map(pre_test_uid)
test_process = test_step.join(b_features).map(process_test).collectAsMap()

#join No map('7xLWtcBooa2op1-wIOtWdQ', (('DoRCeCcJbrsM2BiAKj3trA', 841, 3.65), (47, 4.5)))
#((841, 3.65, 47, 4.5), ('7xLWtcBooa2op1-wIOtWdQ', 'DoRCeCcJbrsM2BiAKj3trA'))
x_test = list(test_process.values())
ids = list(test_process.keys()) #b_id, u_id pair

#########xgb###########
model = xgb.XGBRegressor(n_estimators = 300, max_depth = 6, eval_metric = 'rmse', learning_rate = 0.02, random_state= 100)
model.fit(x_train, y_rate)
y_pred = model.predict(x_test)
print('2nd model finished')
###CF:test_cal, MODEL: y_pred
def final_score(test_cal, y_pred):
    #if the item has a smaller number of neighbors, then the weight of theCF should be smaller
    #if two restaurants both are 4 stars and while the first one has 10 reviews, the second one has 1000 reviews, the average star of the second one is more trustworthy, so the model-based RS score should weigh more
    
    neigbor_dict = dict()
    weight = {}
    for bid, uid in ids:#cal how many users review this business
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
            #model 200 2 0.025
            #neighbor 500: 0.99 0.95 0.9840287044030889
            #neighbor 500: 0.95 0.8  0.9849235061210504
            # count_123: 1620(100+) 38250(20+) 102174(else)
            #0.99 0.95 0.8 rmse 0.9848975344782583 
            #0.99 0.98 0.95 0.9840470370467281
            #0.99 0.98 0.90 0.9840853669416889
            ## count_123: 10576(50+) 29294(20+) 102174  (else)
            #0.99 0.98 0.95
        cf_pred = cf_dict[str(uid)+','+str(bid)]
        
        final_weight = (cf_pred* (1 - model_weight)) + (y_pred[i] *model_weight)
   
        ids_pre[uid+','+bid] = final_weight
    print(' count_123:', count_level1, count_level2, count_level3)
    return ids_pre #{ids: pre}
ids_pre = final_score(test_cal, y_pred)

f_id_keys = list(ids_pre.keys())
f_id_val = list(ids_pre.values())
with open(output_f, 'w') as f:
    f.write("user_id, business_id, prediction\n")
    for i in range(len(y_pred)):
        f.write(str(f_id_keys[i])+','+ str(f_id_val[i]) + '\n')

def test(x):
    data = x.split (',')
    return data[0], data[1], float(data[2])
test_y = testrdd_.map(test).collect()
test_y =  list(test_y)
dict_y = {x[0]+','+x[1]: x[2] for x in test_y} 

dif = 0
n = len(ids_pre)#ids_pre 'LsjYGLrWe6psx1y5m2J-4A,NAZzgDkNIL_DpHg6xu9APQ'
#print('count id pre',n)
#print('count dict_y',len(dict_y))
ids = list(ids_pre.keys())

for key in ids: 
    #print('ids_pre[key] no key error',ids_pre[key] )
    #print('dict_y[key] no key error',dict_y[key] )
    dif += (ids_pre[key]-dict_y[key])**2
avg = dif/n
rmse = avg**0.5
                
print('rmse',rmse)
print('duration:',time.time()- s_time)