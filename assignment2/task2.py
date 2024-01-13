#task2 

'''
Phase 1
1. seperate to chunk 
2. map/run AP algorithm to each chunk （F,1）
3. reduce and output potential candidates
Phase 2 
1. map/find frequent itemsets in each chunk (using random sample algorithm)  (C,v) C是candidate chunk，v是在这个子集中的出现次数
2. reduce and output candidates
'''
import os,sys,time
from pyspark import SparkContext,SparkConf
from collections import Counter
from itertools import combinations
import math,csv

def data_format(data):
    l = []
    for i in data:
        l.append(sorted(i))
    l = sorted(l)
    l.sort(key=lambda t: len(t))
   
    text = ''
    cur_len=1
    for i in l:
        i=tuple(i)
        if len(i)==cur_len+1:
            text += '\n\n'
            cur_len+=1
        if len(i) == cur_len:
            text+= str(i)+','
    return text[:-1].replace (',)', ')').replace (',\n', '\n')

def ap(chunks, s): #in chunks, freq should >=s
    '''AP algorithm 
        Pass 1
        1. calc. n occurance of every single item 
        2. if n>s -> freqent -> candidates
        3. save pass_1_candidates
        Pass 2
        1. calc. m pairs occurance only if they both in pass_1_candidates
        2. if m>s -> freqent -> candidates
        3. save pass_2_candidates'''

    #ex：users[['13', '2', '12', '1', '16'], ['10', '1', '2', '3']]
    #[set({'2'}), set({'1'}), set({'2', '1'})]

    singleton_counter = {} 
    for chunk in chunks: 
        for item in chunk: 
            if item not in singleton_counter: singleton_counter [item] = 0
            singleton_counter [item] += 1

    singletons = [frozenset ([item]) for item in singleton_counter.keys() if singleton_counter [item] >= s] 
    if len (singletons) == 0: return list () 
    
    pass_n_candidates = [singletons]
    
    ####find subsets list(combinations(arr, r))
    while True:
        candidate_counter = {}
        k = len (pass_n_candidates [-1][-1]) + 1
        for p in pass_n_candidates [-1]:
            for q in pass_n_candidates [-1]:
                candidate = p | q
                if len (candidate) != k: continue
                if any ([frozenset(subset) not in pass_n_candidates[-1] for subset in combinations(candidate, k-1)]): continue
                candidate_counter [candidate] = 0
        
        for chunk in chunks:
            for candidate in candidate_counter:
                if candidate.issubset (chunk):
                    candidate_counter [candidate] += 1

        candidates = [candidate for candidate in candidate_counter.keys() if candidate_counter [candidate] >= s] 
        
        if len (candidates) == 0: break
        pass_n_candidates.append (candidates)
    
    res = [] 
    for candidates in pass_n_candidates: 
        res.extend (candidates)
    return res

def translate(data): #delete duplicates
    return list(set(data[1]))

if __name__ == '__main__':
#environment setting
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    conf = SparkConf().setAppName('task1').setMaster('local[*]')
    sc = SparkContext(conf=conf)
    
    #load file and preprocess
    path = sys.argv[3]
    output_path = sys.argv[4]
    threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    
    
    rdd_ = sc.textFile(path).map(lambda x: x.replace('"', '')).map(lambda line: line.split(",")).filter(lambda x: x[0]!='TRANSACTION_DT').cache()

    rdd_ = rdd_.map(lambda x: (x[0]+'-'+x[1], int(x[5])))
    ############################
    with open("customer_product.csv", 'w') as f:
        header = ["DATE-CUSTOMER_ID", "PRODUCT_ID"]
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rdd_.collect(): writer.writerow(row)

    s_time = time.time()
    #################################
    rdd = sc.textFile("customer_product.csv").cache()
    rdd = rdd.map(lambda x: x.split(',')).filter(lambda x: x[0]!='DATE-CUSTOMER_ID')
    rdd = rdd.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b).map(translate).filter(lambda x: len(x) > threshold).persist()
    #ex：businesses[['102', '100', '101', '98'], ['102', '97', '101', '99', '103']]
    rdd_count= rdd.count()

    def phase1(chunk):
        chunk = list(chunk)
        return ap(chunk, math.ceil(support * len(chunk) / rdd_count))

    phase1 = rdd.mapPartitions(phase1).distinct().collect()#.distinct().collect()
    

    with open(output_path,'w') as f:
        f.write("Candidates:\n" )
        text = data_format(phase1)
        f.write(text)
        f.write('\n')

    def phase2(chunks):
        keys= {}
        for chunk in chunks:
            for key in phase1:
                if set(key).issubset(chunk):
                    if key not in keys:
                        keys[key] = 1
                    else:
                        keys[key] += 1

        return keys.items()

    phase2 = rdd.mapPartitions(phase2) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] >= support).map(lambda x: x[0]).collect()

    #print(freqItemset) #[frozenset({'13'}), frozenset({'12'}), frozenset({'12', '13'})]
    with open(output_path,'a') as f:
        f.write('\n')
        f.write("Frequent Itemsets:\n" )
        text = data_format(phase2)
        f.write(text)

    end_time = time.time() - s_time
    print("Duration:",end_time)