
from collections import Counter
from itertools import combinations



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


import time 

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def A_Priori(baskets, support):
    #getting singletons
    all_singles = Counter()
    for basket in baskets:
        for item in basket:
            all_singles[frozenset({item})] += 1
    #get frequent singletons
    currFreq = filter_itemsets(all_singles, support)
    k = 2
    res = currFreq.copy()
    while currFreq:
        #frequent items of length k-1
        currCounter = Counter()
        prevFreq = currFreq
        
        candidates = set()
        for i in prevFreq:
            for j in prevFreq:
                newset = frozenset(list(i) + list(j))
                if len(newset) == k:
                    candidates.add(newset)
                    
        for basket in baskets:
            for candidate in candidates:
                if candidate.issubset(basket):
                    currCounter[candidate] += 1
        currFreq = filter_itemsets(currCounter, support)
        res.extend(list(currFreq))
        k += 1
    return res
        
def filter_itemsets(itemsetCounts, support):
    freq = []
    for item, count in itemsetCounts.items():
        if count >= support:
            freq.append(item)
    return freq
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------



chunks = [{'24'}, {'13'}, {'13'}, {'13', '24'}, {'13', '24'}, {'3', '1'}, {'1', '3', '2'}]*10000000


ts_start = time.time () 
print(ap(chunks,2))
print (time.time() - ts_start)

ts_start = time.time () 
print(A_Priori(chunks,2))
print (time.time() - ts_start)
