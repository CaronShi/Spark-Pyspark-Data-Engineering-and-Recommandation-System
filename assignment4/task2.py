from pyspark import SparkContext,SparkConf
import sys,os
import time
from collections import defaultdict,Counter
from itertools import combinations

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
conf = SparkConf().setAppName('task1').setMaster('local[*]')

sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
path = sys.argv[2]#'../resource/asnlib/publicdata/ub_sample_data.csv' #
output_path1 = sys.argv[3]#'output_task1.txt'#
output_path2 = sys.argv[4]
threshold = int(sys.argv[1])
start = time.time()
rdd = sc.textFile(path)
firstline= rdd.take(1)

rdd1 = rdd.filter(lambda x: x != firstline).map(lambda x: (x.split(',')[0], x.split(',')[1]))
user_buses = rdd1.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b).map(lambda x: (x[0], set(x[1])))#(uid, {group of bids})

user_ids = user_buses.map(lambda x:x[0])
combo = sc.parallelize(combinations(user_ids.collect(), 2))

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
g_rdd_edges = u_threshold.map(lambda x: x[0]).flatMap(lambda x: [x, (x[1], x[0])])
adjacent_uids =g_rdd_edges.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b).map(lambda x: (x[0], set(x[1])))
adjacents_id = adjacent_uids.collectAsMap()
#{uid: {adjacent_uids}}, adjacent_uids['FTZyZbr1eF7s3Ss7uSLcXQ']
def betweenness(nbor):
    b_res = []
    for root in nbor:
        q = [root]
        added = set()
        tree = defaultdict(list) 
        out_n = defaultdict(int) #outdegree
        path_n = defaultdict(int) 
        path_n[root] = 1
        tree_leaf = []
        while len(q)>0:
            new = []
            for node in q:
                if node not in added:
                    added.add(node)
                    for j in nbor[node]:
                        if j not in added and j not in q:
                            path_n[j] += path_n[node]
                            tree[j].append(node)
                            out_n[node] += 1
                            new.append(j)
                    if out_n[node] == 0:
                        tree_leaf.append(node)
            q = new
            
        nc = defaultdict(float) #nodes
        ec = defaultdict(float) #edges
        while len(tree_leaf)>0:
            leaf = tree_leaf.pop(0)
            nc[leaf] += 1
            if tree[leaf]:
                split_cred = nc[leaf] / path_n[leaf]
                for j in tree[leaf]:
                    edges_count = split_cred * path_n[j]
                    ec[frozenset([leaf, j])] += edges_count
                    nc[j] += edges_count
                    out_n[j] -= 1
                    if out_n[j] == 0:
                        tree_leaf.append(j)
        tem_list = []
        for k in ec.keys():
            a = (k,ec[k])
            tem_list.append(a)
        b_res.extend(tem_list)
    
    return b_res

def use_Sort(data):
    return -data[1], data[0][0]

def sorted_edges(data):
    sep_2 = data[1] / 2
    sort = tuple(sorted(list(data[0]))) 
    res = sort, sep_2
    return res

b_rdd = sc.parallelize(betweenness(adjacents_id))
b_res = b_rdd.reduceByKey(lambda x, y: x + y).map(sorted_edges).collect()
        
between_Res = sorted(b_res, key = use_Sort)        
        
with open(output_path1, 'w') as f:
    for e in between_Res:
        f.write(str(e)[1: -1] + "\n")
        
def modularity(d, edge,adjacent):
    visited = set()
    mod = 0
    communit = []
    for node in adjacent:
        if node in visited:
            continue
        q = []
        q.append(node)
        comm = []
        comm.append(node)
        visited.add(node)

        while len(q)>0:
            node = q.pop(0)
            for n in adjacent[node]:
                if n in visited:
                    continue
                else:
                    comm.append(n)
                    visited.add(n)
                    q.append(n)
                    
        communit.append(comm)
        for com in comm:
            degree_i = d[com]
            for j in comm:
                degree_j = d[j]
                a = 0 
                demonitor = len(edge) * 2
                numerator = degree_i * degree_j 
               
                if tuple(sorted((com, j))) in edge:
                    a = 1
     
                mod += a - (numerator/demonitor)
    modularity = mod / (len(edge) * 2)
    return [modularity,communit]
            

def comm_n(adjacent, edges):
    d = {}
    for node in adjacent:
        adjacent_n = len(adjacent[node])
        d[node] = adjacent_n
    max_modularity = modularity(d, edges, adjacent)[0]
    max_comm = modularity(d, edges, adjacent)[1]

    edges_fake = edges.copy()
    while len(edges_fake)>0:
        b_rdd = sc.parallelize(betweenness(adjacent))
        b_res = b_rdd.reduceByKey(lambda x, y: x + y).map(sorted_edges).collect()
        betweenness_ = b_res
        betweenness_.sort(key = lambda x: -x[1])

        maximum_between = []
        maximum_between.append(betweenness_[0][0])
        for edge, between in betweenness_[1:]:
            if between != betweenness_[0][1]:
                break
            else:
                maximum_between.append(edge)
        for edge in maximum_between:
            edges_fake.remove(edge)
            i, j = edge
            adjacent[i].remove(j)
            adjacent[j].remove(i)

        modularity_ = modularity(d, edges,adjacent )[0]
        comm = modularity(d, edges,adjacent )[1]

        if modularity_ > max_modularity:
            max_modularity, max_comm = modularity_, comm
    return max_comm

between_Res_e = set([x[0] for x in between_Res])

communities = []
comm_ns = comm_n(adjacents_id, between_Res_e)
for i in comm_ns:
    communities.append(sorted(i))

def comm_sort(data):
    res = len(data), data[0]
    return res

communities.sort(key = comm_sort )

with open(output_path2, 'w') as file:
    for c in communities:
        ans = str(c)[1: -1]
        file.write( ans +"\n")
       
print(time.time()- start)        
    


   