#task3
import sys
import csv
import time
import random
import binascii
from blackbox import BlackBox
random.seed(553)


if __name__ == '__main__':
    bx = BlackBox()
    start = time.time()
    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]
  
    res = []

    windows = bx.ask(input_path, stream_size)
    res.append([stream_size, windows[0], windows[20], windows[40],
                                windows[60], windows[80]])
    n = stream_size
    i = 0
    while i < num_of_asks-1:
        cur_win = bx.ask(input_path, stream_size)

        for user in cur_win :
            n+=1
            x= stream_size / n
            if random.random() < x:
                p = random.randint(0, stream_size-1)
                windows[p] = user   
            
        res.append([n, windows[0], windows[20], windows[40],
                                windows[60], windows[80]])
        i+=1 
        
    with open(output_path, 'w') as f:
        csv.writer(f).writerow(['seqnum', '0_id', '20_id', '40_id', '60_id', '80_id'])
        for i in range(0, len(res)):
            windows = res[i]
            csv.writer(f).writerow(windows)
    
    print("Duration:", time.time() - start) 