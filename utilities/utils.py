import torch
import numpy as np
import copy
import cvxpy as cp
    



def normal_aggregation(cfg, nets_this_round,fed_avg_freqs,global_w):
    tmp_client_state = copy.deepcopy(global_w)
   
    for key in tmp_client_state:
        tmp_client_state[key] = torch.zeros_like(tmp_client_state[key])

    
    for id in nets_this_round.keys():
        net_para = nets_this_round[id].state_dict()
        for key in tmp_client_state:
            tmp_client_state[key] += net_para[key] * fed_avg_freqs[id]

    
    return tmp_client_state


