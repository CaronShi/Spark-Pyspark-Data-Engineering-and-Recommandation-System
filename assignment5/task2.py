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
   
    for _ in range(n):
        a = random.randint(100, m)
        b= random.randint(100, m)
        res = (a*user_int + b)%m
        hash_res.append(res)
    return hash_res

def FM_func(res):
    N = 8
    n_group = 25

    bx = BlackBox()
    for _ in range(num_of_asks):
        s_users = bx.ask(input_path, stream_size)
       
        R1 = [0] * n

        for u in s_users:
            binary_h = []
            for i in hashs(u):
                binary_h.append(format(i, 'b'))

            for i in range(n):
                n_bin = len(binary_h[i]) - len(binary_h[i].rstrip('0'))
                R1[i] = max(n_bin, R1[i])

        R = list(map(lambda x: 2**x, R1))
        
        grouped_R = []

        for i in range(N):
            start= i * n_group
            end = (i + 1) * n_group
            r = R[start : end]
            grouped_R.append(r)
            
        group_avg = []
        for r in grouped_R:
            avg = sum(r) / n_group    
            group_avg.append(avg)
        grouped_avg = sorted(group_avg)
        res.append((len(set(s_users)), int(grouped_avg[N // 2])))
        
    return res



if __name__ == '__main__':
    bx = BlackBox()
    start = time.time()
    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]
    n = 250
    m = 3444
    FM = FM_func([])
    
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Ground Truth','Estimation'])
        for i, fm in enumerate(FM):
            truth = fm[0]
            est = fm[1]
            writer.writerow((i, truth, est))
    
    print("Duration:", time.time() - start)
    
    
    