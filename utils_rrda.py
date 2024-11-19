import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm 

from sklearn.cluster import KMeans
from torch.utils.data import Dataset

class Random_Dataloader(Dataset):
    def __init__(self, features,labels):
        
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features[idx]
        labels = self.labels[idx]
        return features,labels
    
def get_bad_points(features,classifier,num_class,k_prime,nb_points=1000,num_steps=200):

    with torch.no_grad():
        points = features.clone()

    points.requires_grad = True

    optimizer = torch.optim.Adam([points], lr=0.001)

    # Optimization loop
    for step in range(num_steps):

        optimizer.zero_grad()
        classifier.eval()
        scores = classifier(points)


        std_x = torch.sqrt(points.var(dim=0) + 0.001)

        std_loss = torch.mean(torch.nn.functional.relu(1 - std_x))

        # Calculate class probabilities using softmax
        probabilities = torch.softmax(scores, dim=1)

        # Calculate the entropy for each point
        entropies = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
        loss = - entropies.mean() + std_loss
        # Calculate the gradient of entropy with respect to the points
        loss.backward()
        optimizer.step()
        
    Index = np.argwhere((entropies.detach().cpu().numpy()>0.75*np.log(num_class))) #0.75
    Index = torch.from_numpy(Index).squeeze(1)
    kmeans = KMeans(n_clusters=k_prime).fit(points[Index].cpu().detach().numpy()) 

    return points[Index].detach(),torch.from_numpy((kmeans.labels_ + num_class)).long().cuda() 

def get_good_points(features,classifier,class_desired,num_class,nb_points=1000,num_steps=200):

    with torch.no_grad():
        points = features.clone()

    points.requires_grad = True

    optimizer = torch.optim.Adam([points], lr=0.001) 

    # Optimization loop
    for step in range(num_steps):

        optimizer.zero_grad()
        classifier.eval()
        scores = classifier(points)

        
        std_x = torch.sqrt(points.var(dim=0) + 0.0001)
        
        std_loss = torch.mean(torch.nn.functional.relu(1 - std_x))
        
        # Calculate class probabilities using softmax
        probabilities = torch.softmax(scores, dim=1)
        
        # Calculate the entropy for each point
        entropies = nn.CrossEntropyLoss(reduction='none')(scores,(torch.ones(scores.shape[0])*class_desired).long().cuda()) 
        
        loss = entropies.mean() + std_loss

        loss.backward()
        optimizer.step()

    Index = np.argwhere((entropies.detach().cpu().numpy()<0.25*np.log(num_class)))
    Index = torch.from_numpy(Index).squeeze(1)
    if Index.size(0) > nb_points:
        Index = Index[:nb_points]
        
    return points[Index].detach(),(torch.ones((points[Index]).shape[0])*class_desired).long().cuda()

def compute_h_score_2(args, class_list, gt_label_all, pred_cls_all, open_flag=True):
    
    # class_list:
    #   :source [0, 1, ..., N_share - 1, ...,           N_share + N_src_private - 1]
    #   :target [0, 1, ..., N_share - 1, N_share + N_src_private + N_tar_private -1]
    # gt_label_all [N]
    # pred_cls_all [N, C]
    # open_flag    True/False
    # pred_unc_all [N], if exists. [0~1.0]
    
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros_like(per_class_num)
    pred_label_all = torch.max(pred_cls_all, dim=1)[1] #[N]
    
    if open_flag:
        unc_idx = torch.where(pred_label_all >= args.class_num)[0]
        pred_label_all[unc_idx] = args.class_num # set these pred results to unknown

    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_all == label)[0]
        correct_idx = torch.where(pred_label_all[label_idx] == label)[0]
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))

    per_class_acc = per_class_correct / (per_class_num + 1e-5)

    if open_flag:
        known_acc = per_class_acc[:-1].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    else:
        known_acc = per_class_correct.sum() / (per_class_num.sum() + 1e-5)
        unknown_acc = 0.0
        h_score = 0.0

    return h_score, known_acc, unknown_acc, per_class_acc

def inference(model,dataloader,apply_softmax=True):
    feature_all = []
    pred_all = []
    label_all = []
    model.eval()
    with torch.no_grad():
        for i, (_,inputs, labels,_) in enumerate(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            features,output = model(inputs,apply_softmax)
            
            feature_all.append(features.detach().cpu())
            pred_all.append(output.detach().cpu())
            label_all.append(labels.detach().cpu())
            
    feature_all = (torch.cat(feature_all, dim=0))
    pred_all    = (torch.cat(pred_all, dim=0))
    label_all   = (torch.cat(label_all, dim=0))
    
    return feature_all,pred_all,label_all