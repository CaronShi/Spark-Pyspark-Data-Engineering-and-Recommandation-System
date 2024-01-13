import os,sys,time
from pyspark import SparkContext,SparkConf
import xgboost as xgb #py -3.6 -m pip install xgboost
import pandas as pd #py -3.6 -m pip install pandas 
import numpy as np
import json

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
start = time.time()
conf = SparkConf().setAppName('task2.2').setMaster('local[*]')
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

###input files
output_f ='task2_2_output.csv'
folder_path = 'task2_2_output.csv' #../resource/asnlib/publicdata/yelp_train.csv
test_f = folder_path+'/yelp_val.csv'
train_f = folder_path+'/yelp_train.csv'


s_time = time.time()
rdd_ = sc.textFile(train_f)
header = rdd_.take(1)
rdd_ = rdd_.filter(lambda x: x not in header) #business_id,user_id

def split_data(data):
    data = data.split(',')
    return data[0], (data[1],float(data[2]))

train_rdd = rdd_.map(split_data)#(users_id, business_id, star)

#########test###########
rdd_test = sc.textFile(test_f)
headertest = rdd_test.take(1)
rdd_test = rdd_test.filter(lambda x: x not in headertest) #business_id,user_id

def split_testdata(data):
    data = data.split(',')
    return data[0], data[1]
test_rdd = rdd_test.map(split_testdata)
#########test###########

b_rdd = sc.textFile(folder_path + '/business.json').map(lambda x: json.loads(x))
u_rdd = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x))

b_features = b_rdd.map(lambda x: (x['business_id'], (int(x['review_count']), float(x['stars']))))
u_features = u_rdd.map(lambda x: (x['user_id'], (int(x['review_count']), float(x['average_stars']))))

max_ureview  = u_features.map(lambda x: x[1][0]).max()
max_breview = b_features.map(lambda x: x[1][0]).max()
#min_max scalar(x - min_val) / (max_val - min_val)   
b_features = b_features.map(lambda x: (x[0], (x[1][0]/max_breview, x[1][1]/5)))

u_features = u_features.map(lambda x: (x[0], (x[1][0]/max_ureview, x[1][1]/5)))


#print(b_features.take(5))
#b_rev_max = u_features.collect()
#print(b_rev_max[3])
               
#exit()
        
def preprocess_uid(x):
    u_id = x[0]
    b_id = x[1][0][0]
    rate = x[1][0][1]
    b_rev = x[1][1][0]
    b_star =  x[1][1][1]
    return b_id,(rate,b_rev,b_star)

def preprocess_train(x):#((5.0, 636, 3.97), (69, 3.0))
    rate = x[0][0]
    b_rev = x[0][1]
    b_star =  x[0][2]
    u_rev = x[1][0]
    u_star = x[1][1]
    return (b_rev,b_star,u_rev,u_star),rate


train_step =  train_rdd.leftOuterJoin(u_features).map(preprocess_uid).join(b_features) 
train_process1 = train_step.map(lambda x: x[1]).map(preprocess_train)
train_process =train_process1.collectAsMap()
              
x_train = list(train_process.keys())#(b_rev,b_star,u_rev,u_star)
y_rate = list(train_process.values()) #rating


#########test###########
def pre_test_uid(x):
    u_id = x[0]
    b_id = x[1][0]
    u_rev = x[1][1][0]
    u_star =  x[1][1][1]
    return b_id,(u_id,u_rev,u_star)

def process_test(x):
    u_id = x[0]
    b_id = x[1][0][0]
    u_rev = x[1][1][0]
    u_star =  x[1][1][1]
    b_star = x[1][0][2]
    b_rev = x[1][0][1]
    return (u_id,b_id),(b_rev, b_star,u_rev, u_star)
test_step = test_rdd.join(u_features).map(pre_test_uid)
test_process = test_step.join(b_features).map(process_test).collectAsMap()
 
#join No map('7xLWtcBooa2op1-wIOtWdQ', (('DoRCeCcJbrsM2BiAKj3trA', 841, 3.65), (47, 4.5)))
#((841, 3.65, 47, 4.5), ('7xLWtcBooa2op1-wIOtWdQ', 'DoRCeCcJbrsM2BiAKj3trA'))
x_test = list(test_process.values())
ids = list(test_process.keys())
#########test###########

#########model##########
model = xgb.XGBRegressor(n_estimators = 250, max_depth = 5, eval_metric = 'rmse',learning_rate = 0.025, random_state= 100)
model.fit(x_train, y_rate)
y_pred = model.predict(x_test)

print(y_pred)

with open(output_f, 'w') as f:
    f.write("user_id, business_id, prediction\n")
    for i in range(len(y_pred)):
        f.write(str(ids[i][0])+','+ str(ids[i][1])+','+ str(y_pred[i]) + '\n')
dif = 0
n = len(y_pred)
for i in range(0,n): dif += (y_pred[i]-y_rate[i])**2
avg = dif/n
rmse = avg**0.5

print('rmse',rmse)
print('duration:',time.time()- start)