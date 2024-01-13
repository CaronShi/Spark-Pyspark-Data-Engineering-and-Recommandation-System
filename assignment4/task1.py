#task1

from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession, Row
from graphframes import *
import sys,os
import time
from itertools import combinations
from graphframes import GraphFrame

#os.environ['PYSPARK_PYTHON'] = sys.executable
#os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
#os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell"

sc = SparkContext().getOrCreate()
sc.setLogLevel("ERROR")
spark = SparkSession(sc)
path = sys.argv[2]#'../resource/asnlib/publicdata/ub_sample_data.csv'
output_path = sys.argv[3]#'output_task1.txt'
threshold = int(sys.argv[1])
start = time.time()
rdd = sc.textFile(path)
firstline= rdd.first()
rdd1 = rdd.filter(lambda x: x != firstline).map(lambda x: x.split(','))

user_buses = rdd1.groupByKey().map(lambda x: (x[0], set(x[1])))#[(usr_id, {a group of bus_ids})]

user_ids = user_buses.map(lambda x:x[0])

combo = sc.parallelize(combinations(user_ids.collect(), 2))#5690251

#user_ids_combo = user_ids.cartesian(user_ids).filter(lambda x: x[0][0] != x[1][0]).map(lambda x:frozenset(x)).distinct()
#combo = user_ids_combo.map(tuple)

#[('39FT2Ui8KUXwmUt6hnwy-g', '0FVcoJko1kfZCrJRfssfIA'), ('39FT2Ui8KUXwmUt6hnwy-g', '_6Zg4ukwS0kst9UtkfVw3w')]
def user_combo(x):
    id_bus = x[1]
    return id_bus[0],(x[0],id_bus[1])

def user2_combo(x):
    x1 = x[1]
    i = (x[0], x1[0][0]), x1[0][1].intersection(x1[1])
    return i
    
    
uid_bids1 = combo.leftOuterJoin(user_buses).map(user_combo) #[(uid1, (uid2, {bids})]
uid_bids2 = uid_bids1.leftOuterJoin(user_buses).map(user2_combo)
u_threshold =  uid_bids2.filter(lambda x: len(x[1]) >= threshold)    
g_rdd_edges = u_threshold.map(lambda x: x[0]).flatMap(lambda x: [x, (x[1], x[0])]).toDF(['src', 'dst']) 
g_rdd_nodes = u_threshold.map(lambda x: x[0]).flatMap(lambda x: x).distinct().map(Row('id')).toDF()

gf = GraphFrame(g_rdd_nodes, g_rdd_edges)
g = gf.labelPropagation(maxIter=5)
community = g.rdd.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).map(lambda x: (sorted(x[1])))
community_sort = community.sortBy(lambda x: (len(x), x[0])).collect()

with open(output_path, 'w+') as f:
    for U in community_sort:
        w_in = str(U)[1:-1] + '\n'
        f.write(w_in)
     
    
time = time.time()- start
print("Duration: ",time)