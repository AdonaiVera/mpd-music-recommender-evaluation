import math
import numpy as np
import torch


def get_class(class_divpnt, idx):
    for c in class_divpnt:
        if idx <= c:
            return class_divpnt.index(c)
    return len(class_divpnt)


def get_class_dist(cls_list, num_cls):
    cls_dist = [1e-9] * num_cls
    for i in cls_list:
        if i != -1:
            cls_dist[i]+=1
    return cls_dist


def get_r_precision(answer, cand):
    set_answer = set(answer)
    r = len(set_answer&set(cand[:len(answer)])) / len(answer)
    return r

def get_ndcg(answer, cand):
    cand_len = len(cand) 
    idcg=1
    idcg_idx=2
    dcg=0
    if cand[0] in answer:  dcg=1
    
    for i in range(1,cand_len):
        if cand[i] in answer: 
            dcg += (1/math.log(i+1,2))
            idcg += (1/math.log(idcg_idx,2))
            idcg_idx+=1
    
    return dcg/idcg

def get_rsc(answer, cand):
    cand_len = len(cand)
    for i in range(cand_len):
        if cand[i] in answer:
            return i//10
    return 51

def get_metrics(answer, cand):
    r_precision = get_r_precision(answer, cand)
    ndcg = get_ndcg(answer,cand)
    rsc = get_rsc(answer,cand)
    
    return r_precision, ndcg, rsc

def single_eval(scores, answer, k=10):
    scores = scores.numpy()
    
    # Sort the scores in descending order and get the indices of the top k
    top_k_indices = np.argsort(scores)[::-1][:k]
    
    # Calculate the number of seed tracks that appear in the top k
    relevant_items = set(answer)  # Convert seed to a set for faster lookup
    retrieved_items = set(top_k_indices)  # The indices of the top-k items

    # Calculate the intersection of the retrieved items and the seed tracks
    true_positives = len(relevant_items.intersection(retrieved_items))

    # R-Precision is the proportion of relevant items in the top-k
    r_precision = true_positives / min(k, len(answer)) 
    return r_precision

