import pickle as pkl
import numpy as np
import networkx as nx
import pandas as pd
from operator import itemgetter
import time
from metrics import clustering_metrics
import sys

def GSCUB (segmentation):
    difference = 1
    max_value = 0
    output = []
    previous_max_value = 0
    
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
        
    
    all_candidate_value = []
    edge_index_in_summary = []
    while difference > 0:
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
                objective_value = sum([float(temporary_index_list[i]/(size_list[i]+1)) for i in temporary_list])      
                all_temporary_list[e] = temporary_list[:]
                candidate_value.append((e,objective_value))
            result = sorted(candidate_value,key=itemgetter(1),reverse=True)[0]
            target = result[0]
            output.append(target)
            all_candidate_value.append((output[:],result[1]))
            index_list = temporary_index_list_dic[output[0]]
            edge_index_in_summary = edge_index_in_summary + all_temporary_list[output[0]]
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
                    objective_value = sum([float(temporary_index_list[i]/(size_list[i]+len(output)+1)) for i in set(edge_index_in_summary) | set(temporary_list)]) 
                    candidate_value.append((e,objective_value))
                result = sorted(candidate_value,key=itemgetter(1),reverse=True)[0]
                output.append(result[0])
                all_candidate_value.append((output[:],result[1]))
                index_list = temporary_index_list_dic[output[-1]]
                edge_index_in_summary = edge_index_in_summary + all_temporary_list[output[-1]]
            else:
                difference  = -1

    return float(sorted(all_candidate_value,key=itemgetter(1),reverse=True)[0][1]), sorted(sorted(all_candidate_value,key=itemgetter(1),reverse=True)[0][0])



def prepare_ksegments(series):
    '''
    '''
    N = len(series)

    dists = np.zeros((N,N))
            
    mean_dict = {}    

    for i in range(N):
        mean_dict[(i,i)] = [i for i in series[i]]

    for i in range(N):
        for j in range(N-i):
            r = i+j
            if i != r:
                sub_segment = series[i:r+1]
                error, representative = GSCUB(sub_segment)

                mean_dict[(i,r)] = representative
                dists[i][r] = error
                
    return dists, mean_dict

def k_segments(series, k):
    '''
    '''
    N = len(series)

    k = int(k)

    dists, means = prepare_ksegments(series)

    k_seg_dist = np.zeros((k,N+1))
    k_seg_path = np.zeros((k,N))
    k_seg_dist[0,1:] = dists[0,:]

    for i in range(k):
        k_seg_path[i,:] = i

    for i in range(1,k):
        for j in range(i,N):
            choices = k_seg_dist[i-1, :j] + dists[:j, j]
            best_index = np.argmax(choices)
            best_val = np.max(choices)

            k_seg_path[i,j] = best_index
            k_seg_dist[i,j+1] = best_val


    reg = np.zeros(N)
    rhs = len(reg)-1
    range_list = []
    for i in reversed(range(k)):
        if i == k-1:
            lhs = k_seg_path[i,rhs]
            range_list.append((int(lhs),int(rhs+1)))
            reg[int(lhs):rhs+1] = int(i)
            rhs = int(lhs)
        else:
            lhs = k_seg_path[i,rhs]
            range_list.append((int(lhs),int(rhs)))
            reg[int(lhs):rhs] = int(i)
            rhs = int(lhs)   
            
    c_list = []
    for i in range_list:
        c_list.append(i[0])
        c_list.append(i[1])
        
    c_list = sorted(set(c_list))
    c_list = c_list[1:-1]
    

    means_index = []
    for i,j in enumerate(sorted(c_list)):
        if i == 0:
            means_index.append((0,j-1))
        else:
            means_index.append((sorted(c_list)[i-1],j-1))
            
    if N > 1:
        means_index.append((sorted(c_list)[-1],N-1))  

    summary_graphs = {}

    for i in means_index:
        summary_graphs[i] = means[i]

    return list([int(i) for i in reg]), summary_graphs


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

if __name__ == '__main__':

    data_path = sys.argv[1]

    output_path = sys.argv[2]

    true_lables_path = sys.argv[3]

    k = int(sys.argv[4])

    with open (data_path, 'rb') as fp:
        ego_list = pkl.load(fp)


    with open(true_lables_path) as f:
        true_lables = [int(line.rstrip('\n')) for line in f]

    true_cut_points = []
    for i,j in enumerate(true_lables):
        if i != len(true_lables)-1:
            if true_lables[i] != true_lables[i+1]:
                true_cut_points.append(i)

    ego_list,edges_dic = pre_processing(ego_list)

    reversed_edges_dic = dict(map(reversed, edges_dic.items()))

    start = time.time()
    segmentation,summary_graphs = k_segments(ego_list,k)
    end = time.time()

    cm = clustering_metrics(true_lables, segmentation)
    acc, nmi, f1_macro = cm.evaluationClusterModelFromLabel()


    with open(output_path,'a') as f:
        f.write('time is : ' +  str(end - start) + '\n')
        f.write('segmentation is : ' + str(segmentation) + '\n')
        f.write('acc is : ' + str(acc) + '\n')
        f.write('nmi is : ' + str(nmi) + '\n')
        f.write('f1_macro is : ' + str(f1_macro) + '\n')
        for i in summary_graphs.keys():
            f.write('summary graph ' + str(i) + ' is ' + str([reversed_edges_dic[e] for e in summary_graphs[i]])   + '\n')
