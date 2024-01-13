#assignment 5
import math
import sys
import csv
import time
import random
import binascii
from blackbox import BlackBox
random.seed = 553

def hashs(user):
    user_int = int(binascii.hexlify(user.encode('utf8')),16)
    hash_res = []
   
    for _ in range(k):
        a = random.randint(100, 69997)
        b= random.randint(100, 69997)
        res = (a*user_int + b)%m
        hash_res.append(res)
    return hash_res

def FPR():

    fpr = []
    users_set = set()
    filter_l = []
    for _ in range(m):
        filter_l.append(0)

    for _ in range(num_of_asks):
        s_users = bx.ask(sys.argv[1], stream_size)
        visited = set()
        filter = []
        FP = 0
        for user in s_users:
            if user in users_set:
                visited.add(user)
            hashs = hashs(user)

            filter.append(all(filter_l[hash] == 1 for hash in hashs))
            for hash in hashs:
                filter_l[hash] = 1

            if user in visited and not filter[s_users.index(user)]:
                FP += 1  
        FPR = FP / stream_size
        users_set = users_set.union(s_users)
        fpr.append(FPR)
    return fpr

if __name__ == '__main__':
    bx = BlackBox()
    start = time.time()
    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]
    
    m = 69997
    k = int(m/(stream_size*num_of_asks)*math.log(2))
    FPR_l = FPR()

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'FPR'])
        for i, fpr in enumerate(FPR_l):
            writer.writerow((i, fpr))

    
    print("Duration:", time.time() - start)
    