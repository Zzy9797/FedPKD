import copy
import math
import random
import time

import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import sys
sys.path[0]=''
from utilities.config import get_args
from utilities.utils import normal_aggregation
import datetime
import logging
from utilities.evaluate import compute_acc, compute_local_test_accuracy
from utilities.model import simplecnn,fashioncnn, resnet18_fmnist, resnet18
from utilities.prepare_data import get_dataloader
from utilities.attack import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def local_train_pfedgraph(args, round, nets_this_round,teachers_this_round, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list, student_losses_firstbatch, teacher_losses_firstbatch, rounds, ratios):
    
    criterion = torch.nn.CrossEntropyLoss()
    for net_id, net in nets_this_round.items():
        teacher = teachers_this_round[net_id]
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)  

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
            optimizer_teacher = optim.Adam(filter(lambda p: p.requires_grad, teacher.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
            optimizer_teacher = optim.Adam(filter(lambda p: p.requires_grad, teacher.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
            optimizer_teacher = optim.SGD(filter(lambda p: p.requires_grad, teacher.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        
        
        kl=nn.KLDivLoss(reduction="batchmean").cuda()

        
        net.cuda()
        net.train()
        teacher.cuda()
        teacher.train()

        
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)   
                x, target = next(iterator)
            x, target = x.cuda(), target.cuda()
            teacher_out = teacher(x)
            
            optimizer.zero_grad()
            optimizer_teacher.zero_grad()
            target = target.long()

            out = net(x)
            if iteration == 0:
                student_firstbatch = criterion(out,target)
                statistic_student_firstbatch = student_firstbatch.item()
                teacher_firstbatch = criterion(teacher_out,target)
                statistic_teacher_firstbatch = teacher_firstbatch.item()
                ratio = statistic_teacher_firstbatch/statistic_student_firstbatch
                if round>0:
                    if (ratio/ratios[net_id][-1])<args.ratio:
                        teacher_out=teacher_out.detach()
                        loss=student_firstbatch+args.alpha3*kl(F.log_softmax(out, dim=1),F.softmax(teacher_out, dim=1))
                        loss.backward()
                        optimizer.step()
                    else:
                        out=out.detach()
                        loss=teacher_firstbatch+args.alpha3*kl(F.log_softmax(teacher_out, dim=1),F.softmax(out, dim=1))
                        loss.backward()
                        optimizer_teacher.step()

                else:
                    loss=student_firstbatch
                    loss.backward()
                    optimizer.step()

            else:
                if round >0:
                    if (ratio/ratios[net_id][-1])<args.ratio:
                        teacher_out = teacher_out.detach()
                        loss = criterion(out, target)+args.alpha3*kl(F.log_softmax(out, dim=1),F.softmax(teacher_out, dim=1))
                        loss.backward()
                        optimizer.step()
                    else:
                        out = out.detach()
                        loss = criterion(teacher_out, target)+args.alpha3*kl(F.log_softmax(teacher_out, dim=1),F.softmax(out, dim=1))
                        loss.backward()
                        optimizer_teacher.step()
                else:
                    loss = criterion(out,target) 
                    loss.backward()
                    optimizer.step()
                
        if round>0:
            if (ratio/ratios[net_id][-1])>=args.ratio:
                net.load_state_dict(teacher.state_dict())


        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        net.to('cpu')
        teacher.to('cpu')
        student_losses_firstbatch[net_id].append(statistic_student_firstbatch)
        teacher_losses_firstbatch[net_id].append(statistic_teacher_firstbatch)
        ratios[net_id].append(ratio)

    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()

args, cfg = get_args()
args.model = args.backbone
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

n_party_per_round = int(args.n_parties * args.sample_fraction)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

benign_client_list = random.sample(party_list, int(args.n_parties * (1-args.attack_ratio)))
benign_client_list.sort()
print(f'>> -------- Benign clients: {benign_client_list} --------')

train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)

if args.dataset != 'yahoo_answers' and args.backbone == 'resnet18':
    if args.dataset == 'fashionMnist':
        model = resnet18_fmnist
    else:
        model = resnet18
else:
    if args.dataset == 'cifar10':
        model = simplecnn
    elif args.dataset == 'cifar100':
        model = simplecnn
    elif args.dataset=='fashionMnist':
        model= fashioncnn   
    elif args.dataset=='svhn':
        model= simplecnn
      
                    
    
global_model = model(cfg['classes_size'])
global_parameters = global_model.state_dict()  
if args.backbone == 'resnet18':
    global_parameters = {key: global_parameters[key].float() for key in global_parameters}
global_model_teacher = model(cfg['classes_size'])
global_parameters_teacher = global_model_teacher.state_dict()  
local_models = []
local_models_teacher = []
best_val_acc_list, best_test_acc_list = [],[]
dw = []
student_losses_firstbatch = []
teacher_losses_firstbatch = []
rounds = []
rounds.append(0)
ratios = []
accs=[]
accs.append(0)

for i in range(cfg['client_num']):
    local_models.append(model(cfg['classes_size']))
    local_models_teacher.append(model(cfg['classes_size']))
    dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)

    student_losses_firstbatch.append([])
    teacher_losses_firstbatch.append([])    
    ratios.append([])


for net in local_models:
    net.load_state_dict(global_parameters)

for teacher in local_models_teacher:
    teacher.load_state_dict(global_parameters_teacher)
    
for round in range(cfg["comm_round"]):
    rounds.append(round+1)
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction < 1.0:
        print(f'>> Clients in this round : {party_list_this_round}')
    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    teachers_this_round = {k: local_models_teacher[k] for k in party_list_this_round}

    for key,teacher in teachers_this_round.items():
        teacher.load_state_dict(nets_this_round[key].state_dict())
        nets_this_round[key].load_state_dict(global_parameters)

    distributions_this_round={k:data_distributions[k] for k in party_list_this_round}
    nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}
    teachers_param_start = {k: copy.deepcopy(local_models_teacher[k]) for k in party_list_this_round}
    mean_personalized_acc = local_train_pfedgraph(args, round, nets_this_round, teachers_this_round, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list, student_losses_firstbatch, teacher_losses_firstbatch, rounds, ratios)
    accs.append(mean_personalized_acc)
    total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
    fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round}
    
    if args.attack_type != 'None':
        manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)
        manipulate_gradient(args, None, teachers_this_round, benign_client_list, teachers_param_start)
                                           

    global_parameters = normal_aggregation(cfg, nets_this_round, fed_avg_freqs,global_parameters)


    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)