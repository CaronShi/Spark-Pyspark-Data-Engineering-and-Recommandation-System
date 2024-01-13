import sys
import math
import time,random
import numpy as np
from pyspark import SparkContext
from sklearn.cluster import KMeans
'''
#TODO
seperate initialize_sets(cls, data) into two different functions get result_dic, get_assigned seperately

change keys() to items()

change get_cs_from_rs(rs_id, n_clusters)

change  
    ds = dict()
    rs = dict()
    cs = dict()

change output functions
'''

def K_mean(n_cluster, data, id):
        kmeans = KMeans(n_clusters = n_cluster).fit(data)
        label = kmeans.labels_
        cls_rdd = sc.parallelize(zip(label, id)) \
            .groupByKey() \
            .mapValues(list)
        return cls_rdd

def initialize_sets(cls, data): 
    get_assigned = set()
    result_dic = dict()
    for c_idx in cls.keys():
        id_list = cls[c_idx]
        vals = []
        for point in id_list:
            vals.append(data[point])
        vals = np.array(vals)

        n = len(vals)
        sums = vals.sum(axis=0)
        sum_sq = (vals ** 2).sum(axis=0)

        if n <= 1:
            for need_id in id_list:
                rs[need_id] = data[need_id]
        else:
            result_dic[c_idx] = [id_list, n, sums, sum_sq]

            get_assigned = get_assigned.union(set(id_list))

    return result_dic, get_assigned


def get_cs_from_rs(rs_id, n_clusters):
    if len(rs_id) >= n_clusters:
        values = [rs[id] for id in rs_id]
        cls_rdd = K_mean(n_clusters, values, rs_id) #clusters**
        cls = cls_rdd.collectAsMap()
        new_cs, remove_list = initialize_sets(cls, rs)

        new_key = len(cs)
        for k in new_cs.keys():
            cs[new_key] = new_cs[k]
            new_key += 1
    else: return 
    return remove_list

def rs_pop_out(cs_id):
    for id in cs_id or []:
        rs.pop(id)

def assign(ds, cluster_assign, k, v):
        ds[cluster_assign][0].append(k)
        ds[cluster_assign][1] += 1
        ds[cluster_assign][2] += v
        ds[cluster_assign][3] += v ** 2

def mah_dist(cls, values):
    n = cls[1]
    sums = np.array(cls[2])
    sum_sq = np.array(cls[3])

    variance = sum_sq / n
    std = np.sqrt(variance - np.square(sums / n))

    step2 = (values - (sums / n)) / std
    step1 = np.square(step2)
    dist = np.sqrt(np.sum(step1))

    return dist


if __name__ == '__main__':
    start_time = time.time()

    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]

    sc = SparkContext().getOrCreate()
    sc.setLogLevel("ERROR")

    rdd1 = sc.textFile(input_file).map(lambda x: x.split(',')) #**
    rdd = rdd1.map(lambda x: (int(x[0]), list(float(num) for num in x[2:]))).collectAsMap()
    ids = list(rdd.keys())
    n_data = len(rdd)
    # step 1: load 20% randomly
    
    random.shuffle(ids)

    sample_id = ids[:int(0.2 * n_data)]
    curr_index = ids[int(0.2 * n_data):]

    sample_data = [rdd[i] for i in sample_id]
    sample = dict(zip(sample_id, sample_data))

    # step 2: run k mean k*5
    
    ds = dict()
    rs = dict()
    cs = dict()
    cls_rdd = K_mean(n_cluster*5,sample_data, sample_id)
    

    # step 3: move outliers to rs
    
    
    ds_id = cls_rdd.filter(lambda x: len(x[1]) > 1).flatMap(lambda x: x[1]).collect()
    rs_id = cls_rdd.filter(lambda x: len(x[1]) == 1).flatMap(lambda x: x[1]).collect()

    for id in rs_id:
        rs[id] = sample[id]

    # step 4: run kmean with ds
    values = [sample[id] for id in ds_id]
   
    cls_rdd = K_mean(n_cluster, values, ds_id)

    init_cls = cls_rdd.collectAsMap()

    # step 5: generate the DS clusters
    ds, _ = initialize_sets(init_cls, sample)

    # step 6: generate cs andrs with rs
    
    rs_pop_out(get_cs_from_rs(rs_id, n_cluster * 5))
  
    n_ds = 0
    for cls in ds:
        n_ds += ds[cls][1]

    n_cs = 0
    for cls in cs:
        n_cs += cs[cls][1]
    
    inter_res = [[n_ds, len(cs), n_cs, len(rs)]]

    for _ in range(0,4):
        sample_id = curr_index[:int(0.2 * n_data)]
        curr_index = curr_index[int(0.2 * n_data):]
        sample_data = [rdd[i] for i in sample_id]
        sample_i = dict(zip(sample_id, sample_data))
        
        for point, val in sample_i.items():
            min_dist = math.inf
            cluster_assign = None
            values = np.array(val)
            d = len(values)
            theta = 2 *  d ** 0.5

            for k in ds.keys():
                distance = mah_dist(ds[k], values)
                if distance < min_dist:
                    min_dist = distance
                    cls_id = k
            
            if min_dist <= theta:
                assign(ds, cls_id, point, values)
                # print(discard_glb[cluster_assign][1:])
            else:
                for k in cs.keys():
                    distance = mah_dist(cs[k], values)
                    if distance < min_dist:
                        min_dist = distance
                        cls_id = k
                if min_dist <= theta:
                    assign(cs, cls_id, point, values)
                    # print(discard_glb[cluster_assign][1:])
                else:
                    rs[point] = val
            
        # step 11: generate cs and rs with rs
        rs_id = rs.keys()
        rs_pop_out(get_cs_from_rs(rs_id, n_cluster * 5))

        n_ds = 0
        for cls in ds:
            n_ds += ds[cls][1]

        n_cs = 0
        for cls in cs:
            n_cs += cs[cls][1]
        inter_step = [n_ds, len(cs), n_cs, len(rs)]
        inter_res.append(inter_step)
       
        
    res = sorted(
        [(point, label) for label in ds for point in ds[label][0]] + 
        [(point, -1) for label in cs for point in cs[label][0]] + 
        [(point, -1) for point in rs] )

    with open(output_file, 'w') as out:
        out.write('The intermediate results:\n')
        for i in range(len(inter_res)):
            x = inter_res[i]
            out.write("Round {}: {},{},{},{}\n".format(i+1, x[0],x[1], x[2],x[3]))
        out.write('\n')
        out.write('The clustering results:\n')
        for i in res:
            out.write("{},{}\n".format(i[0], i[1]))

    print("Duration:", time.time() - start_time)