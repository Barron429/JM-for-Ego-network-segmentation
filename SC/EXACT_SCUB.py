import pickle as pkl
import numpy as np
import networkx as nx
import pandas as pd
from operator import itemgetter
import time
from metrics import clustering_metrics
import sys

def FIXED_SIZE_GREEDY_SCUB (segmentation,M,index_list,size_list,index_dic_list,all_edges,iteration):

    difference = 0
    output = []
    
    all_candidate_value = []
    edge_index_in_summary = []
    while difference < iteration:
        candidate_value = []
        if len(output) == 0:
            temporary_index_list_dic = {}
            all_temporary_list = {}
            for e in all_edges:
                temporary_list = []
                temporary_index_list = index_list[:] # make a new copy
                for i,j in enumerate(index_dic_list):
                    if e in j:
                        temporary_index_list[i] = temporary_index_list[i] + 1
                        temporary_list.append(i)
                temporary_index_list_dic[e] = temporary_index_list[:]      
                objective_value = sum([float(temporary_index_list[i]/(size_list[i]+iteration)) for i in temporary_list])      
                all_temporary_list[e] = temporary_list[:]
                candidate_value.append((e,objective_value))
            result = sorted(candidate_value,key=itemgetter(1),reverse=True)[0]
            target = result[0]
            output.append(target)
            all_candidate_value.append((output[:],result[1]))
            index_list = temporary_index_list_dic[output[0]]
            edge_index_in_summary = edge_index_in_summary + all_temporary_list[output[0]]
            difference = difference + 1
        else:
            candidate_elements = all_edges - set(output)
            candidate_value = []
            if len(candidate_elements) > 0:
                temporary_index_list_dic = {}
                all_temporary_list = {}
                for e in candidate_elements:
                    temporary_list = []
                    temporary_index_list = index_list[:]
                    for i,j in enumerate(index_dic_list):
                        if e in j:
                            temporary_index_list[i] = temporary_index_list[i] + 1
                            temporary_list.append(i)
                    temporary_index_list_dic[e] = temporary_index_list[:] 
                    all_temporary_list[e] = temporary_list[:]
                    objective_value = sum([float(temporary_index_list[i]/(size_list[i]+iteration)) for i in set(edge_index_in_summary) | set(temporary_list)]) 
                    candidate_value.append((e,objective_value))
                result = sorted(candidate_value,key=itemgetter(1),reverse=True)[0]
                output.append(result[0])
                all_candidate_value.append((output[:],result[1]))
                index_list = temporary_index_list_dic[output[-1]]
                edge_index_in_summary = edge_index_in_summary + all_temporary_list[output[-1]]
                
                difference = difference + 1

    return float(all_candidate_value[-1][1]), all_candidate_value[-1][0]

def EXACT_SCUB(segmentation):

    EXACT_SCUB_list = [] 

    M = len(segmentation)
    
    index_list = [0] * M
    
    size_list = [len(graph) for graph in segmentation]
    
    index_dic_list = [{}]* M
    
    all_edges = []
    for j,g in enumerate(segmentation):
        ege_dic = {}
        for e in g:
            ege_dic[e] = j
            all_edges.append(e)
        index_dic_list[j] = ege_dic

    all_edges = set(all_edges)

    for i in range(1,len(all_edges)+1):
        EXACT_SCUB_list.append(FIXED_SIZE_GREEDY_SCUB(segmentation,M,index_list,size_list,index_dic_list,all_edges,i))

    return sorted(EXACT_SCUB_list,key=itemgetter(0),reverse=True)[0][0], list(sorted(EXACT_SCUB_list,key=itemgetter(0),reverse=True)[0][1])


def pre_processing(ego_list):
    graph_list = []
    edges_dic = {}
    for g in ego_list:
        graph = []
        for e in g.edges:
            if tuple(sorted(e)) not in edges_dic:
                edges_dic[tuple(sorted(e))] = len(edges_dic)
            graph.append(edges_dic[tuple(sorted(e))])
        graph_list.append(graph)
    return graph_list,edges_dic

def jaccard_distance_list(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return 1 - float(intersection) / union

def compute_JD_list(sequence,representative):
    JD = 0
    edge_set = [list(g) for g in sequence]
    for g in edge_set:
        JD = JD + jaccard_distance_list(g, representative) 
    return JD

if __name__ == '__main__':

    data_path = sys.argv[1]

    output_path = sys.argv[2]

    with open (data_path, 'rb') as fp:
        ego_list = pkl.load(fp)

    ego_list,edges_dic = pre_processing(ego_list)

    start = time.time()
    V, summary = EXACT_SCUB(ego_list)
    end = time.time()

    JD = compute_JD_list(ego_list,summary)

    with open(output_path,'a') as f:
        f.write('time is : ' +  str(end - start) + '\n')
        f.write('JD is : ' +  str(JD) + '\n')
        f.write('summary graph is ' + str(summary) + '\n')