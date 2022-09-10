# Corresponding paper
This project corresponds to the paper

Jaccard Median for Ego-network Segmentation, ICDM, 2022

Haodi Zhong, Grigorios Loukides, Alessio Conte and Solon P. Pissis

## Author of code

Haodi Zhong

If you have issues, please email:

zhonghaodi429@gmail.com

## Dependency


We used python (3.8.3) and  Gurobi (9.5.1) to implement our algorithms. Our implementation was tested on Windows 10.

- numpy
- networkx 
- pandas 
- munkres
- scikit-learn

## The dataset is publicly available at:

https://projects.csail.mit.edu/dnd/DBLP/ 

## Data information

For DBLP, we created 5 sequences (ego_list_1.pkl...ego_list_5.pkl). \
Each sequence has a ground truth (label_1.txt...label_5.txt) 

## How to run the segmeatation algorithms

The implemented algorithms are: 

* DP_EXACT_SC for DP-EXACT_{SC} (D-ESC)	      [Dynamic programming and EXACT-SC] 	
* DP_EXACT_SCUB for DP-EXACT-SC_{UB} (D-ESC_UB) [Dynamic programming and EXACT-SC_UB] 
* DP_G_SCUB for DP-GSC_{UB} (D-GSC_UB)	      [Dynamic programming and GREEDY-SC_UB] 	 
* TD_EXACT_SCUB for TD_EXACT-SC{UB} (T-ESC_UB)  [Top-down and EXACT-SC_UB] 	 
* TD_G_SCUB for TD-GSC_{UB} (T-GSC_UB)          [Top-down and GREEDY-SC_UB]	


#

Any of these algorithms can be executed with: 	 
	 
python Program_Name.py ./data/Data_name/ego_list_1.pkl ./data/Data_name/Output.txt  ./data/Data_name/label_1.txt k	 

--- Program_name.py is the name of the algorithm. It can be DP_EXACT_SC, DP_EXACT_SCUB, DP_G_SCUB, TD_EXACT_SCUB, or TD_G_SCUB	 \
--- ./data/Data_name is the path. Data_name is DBLP	 \
--- ego_list_1.pk1 is the first ego-network sequence. Instead of 1, it could be 2,3,4, or 5 for the other ego-network sequences.	 \
--- Output.txt corresponds to the output. The output is displayed on the screen and includes runtime of algorithm, segmeatation results, segmeatation performances (ACC, NMI, F score) and the summary graph of each segment. 	 \
--- label_1.txt is the file containing the ground truth for the first ego-network sequences. Instead of 1, it could be 2,3,4, or 5 for the ground truth of the other ego-network sequences.	 \
--- k is the desired number of segments	 

#

* Example 1:	 

python DP_G_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/Output.txt  ./data/dblp/label_1.txt 5	 
	 
* Example 2: 	 

To execute DP_EXACT_SC on the toy data in Fig1 of our paper:	 \
python DP_EXACT_SC.py 	 

## How to run JM algorithms

First, please go to SC folder.

The implemented algorithms are:

* EXACT_SC for EXACT-SC  
* EXACT_SCUB for EXACT-SC_UB
* G_SCUB for GREEDY-SC_UB
* JMA for 2-approximation algorithm 

Any of these algorithms(except EXACT_SC ) can be executed with: 

python Program_Name.py ../data/Data_name/ego_list_1.pkl 

For EXACT_SC, please run:
python EXACT_SC.py

## The source codes of the methods we compare against

COPOD: https://github.com/winstonll/COPOD \
ECOD: https://github.com/yzhao062/pyod/blob/master/pyod/models/ecod.py \
ROD: https://codeocean.com/capsule/2686787/tree/v2 \
SNAPNETS: http://github.com/SorourAmiri/SnapNETS 
