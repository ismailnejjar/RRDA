import os
import torch 
import shutil 
import numpy as np

from tqdm import tqdm 
from model.SFUniDA import SFUniDA 
from model.SFUniDA import Classifier 

from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader
from scipy.spatial.distance import cdist

from config.model_config import build_args
from utils.net_utils import set_logger, set_random_seed
from utils.net_utils import compute_h_score, Entropy
import torch.nn.functional as F

from utils_rrda import *

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def get_synthetic_points(args, model, dataloader, nb_points):
    
    feature_all,_,_ = inference(model, dataloader)
    get_bad_features = []
    sample_data     = {}
    
    for i in range(args.class_num):
        features,classe = get_good_points(feature_all.cuda().float(),model.class_layer,class_desired = i,num_class = args.class_num,nb_points=nb_points,num_steps=1000)
        sample_data['Class '+str(i)] = features.shape[0]
        if(i == 0):
            new_features = features
            new_labels = classe
        else:
            new_features = torch.cat((new_features,features),0)
            new_labels = torch.cat((new_labels,classe),0)
    
    mean = np.repeat(feature_all.cpu().numpy(), 2, axis=0) #before 2 
    noise = np.random.normal(loc=0, scale=2.0, size=mean.shape)
    features = mean + noise
    
    outlier_feature,outlier_labels = get_bad_points(torch.from_numpy(features).cuda().float(),model.class_layer,args.class_num ,args.k_prime_classes,nb_points=nb_points,num_steps=1000) 

    sample_data['Outlier'] = outlier_feature.shape[0]
    
    new_features = torch.cat((new_features,outlier_feature),0)
    new_labels = torch.cat((new_labels,outlier_labels),0)
    print(new_features.shape)
    print('Number of generated sample for each class', sample_data)
    
    return new_features.cpu(), new_labels.cpu()

def train_new_classifier(args,model,features,labels):
    clasifier_dataset = Random_Dataloader(features,labels)
    clasifier_loader = torch.utils.data.DataLoader(clasifier_dataset, batch_size=2*args.batch_size, shuffle=True)

    model_c_t = Classifier(embed_dim=args.embed_feat_dim, class_num = args.class_num + args.k_prime_classes, type='wn').cuda()
    
    with torch.no_grad():
        model_c_t.fc.weight[:args.class_num ,:] = model.class_layer.fc.weight
        model_c_t.fc.weight_v[:args.class_num ,:] = model.class_layer.fc.weight_v
        model_c_t.fc.weight_g[:args.class_num ,:] = model.class_layer.fc.weight_g

        model_c_t.fc.bias[:args.class_num ] = model.class_layer.fc.bias

    optimizer_c = torch.optim.SGD(model_c_t.parameters(), lr=1e-2,momentum=0.9, weight_decay=1e-3,nesterov=True)

    for e in (range(50+1)):
        loss_running = 0
        for i, (inputs, labels) in enumerate(clasifier_loader):

            model_c_t.train()

            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer_c.zero_grad()

            outputs = model_c_t(inputs)
            loss = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs,labels.long())
            loss_running  = loss_running  + loss.item()
            loss.backward()
            optimizer_c.step()
            
        if(e%25 == 0):
            print('Epoch ',e,' loss : ',loss_running/len(clasifier_loader))
            
    return model_c_t

    
def train(args, model, old_classifier, train_dataloader, test_dataloader, optimizer,optimizer_c,score_bank,feature_bank, epoch_idx=0.0):
        
    all_pred_loss_stack = []
    iter_idx = epoch_idx * len(train_dataloader)
    iter_max = args.epochs * len(train_dataloader)
    
    model.train()
    for imgs_train, _, _, imgs_idx in tqdm(train_dataloader, ncols=60):
        iter_idx += 1
        imgs_idx = imgs_idx.cuda()
        imgs_train = imgs_train.cuda()
        
        features_test, outputs_test  = model(imgs_train,apply_softmax=False)
        softmax_out  = torch.softmax(outputs_test , dim=1)
        
        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.detach().clone()

            pred_bs = softmax_out

            feature_bank[imgs_idx] = output_f_.detach().clone().cuda()
            score_bank[imgs_idx] = softmax_out.detach().clone().cuda()

            distance = output_f_ @ feature_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k= 5  + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C
            
            
        alpha = 1  #(1 + 10 * iter_idx / iter_max) ** (-5) 

        softmax_out_un = softmax_out.unsqueeze(1).expand( -1, 5, -1)  # batch x K x C

        loss = torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)
        ) # Equal to dot product

        mask = torch.ones((imgs_train.shape[0], imgs_train.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = softmax_out.T  # .detach().clone()#

        dot_neg = softmax_out @ copy  # batch x batch

        dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)
        loss += neg_pred * alpha
        
        lr_scheduler(optimizer, iter_idx, iter_max) 
        lr_scheduler(optimizer_c, iter_idx, iter_max) 
        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()
        
        all_pred_loss_stack.append(loss.cpu().item())
        
    train_loss_dict = {}
    train_loss_dict["all_pred_loss"] = np.mean(all_pred_loss_stack)
    return train_loss_dict, score_bank, feature_bank

@torch.no_grad()
def test(args, model, dataloader, src_flg=False):
    
    model.eval()
    gt_label_stack = []
    pred_cls_stack = []
    
    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0
    
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda() 
        _, pred_cls = model(imgs_test, apply_softmax=True)
        gt_label_stack.append(imgs_label)
        pred_cls_stack.append(pred_cls.cpu())
    
    gt_label_all = torch.cat(gt_label_stack, dim=0) #[N]
    pred_cls_all = torch.cat(pred_cls_stack, dim=0) #[N, C]

    h_score, known_acc, unknown_acc, _ = compute_h_score_2(args, class_list, gt_label_all, pred_cls_all, open_flg)
    return h_score, known_acc, unknown_acc

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    this_dir = os.path.abspath('') 
    model = SFUniDA(args)

    model = model.cuda()
    

    save_dir = os.path.join(this_dir, "checkpoints_ismail_aad", args.dataset,"s_{}_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type,
                           "ablation_{}".format(args.k_prime_classes))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    args.logger = set_logger(args, log_name="log_target_training.txt")

    if args.reset:
        raise ValueError

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(args.checkpoint)
        raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")

    shutil.copy("./train_target.py", os.path.join(args.save_dir, "train_target.py"))
    shutil.copy("./utils/net_utils.py", os.path.join(args.save_dir, "net_utils.py"))

    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)

    target_train_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True)
    target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size*2, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False)

    notation_str =  "\n=======================================================\n"
    notation_str += "   START TRAINING ON THE TARGET:{} BASED ON SOURCE:{}  \n".format(args.t_idx, args.s_idx)
    notation_str += "======================================================="

    args.logger.info(notation_str)
   
    if(args.dataset == 'VisDA'):
        nb_points = 10000
    else:
        nb_points = 1000
        

    synthetic_features,synthetic_labels = get_synthetic_points(args, model, target_test_dataloader,nb_points)

    old_classifier = model.class_layer

    model.class_layer = train_new_classifier(args,model,synthetic_features,synthetic_labels)

    hscore, knownacc, unknownacc = test(args, model, target_test_dataloader, src_flg=False)

    args.logger.info("Current: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(hscore, knownacc, unknownacc))

    param_group = []
    param_group_c = []
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]

    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    for k, v in model.class_layer.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr}]

    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    optimizer_c = torch.optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    best_h_score = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0
    best_epoch_idx = 0

    num_sample = len(target_test_dataloader.dataset)

    score_bank = torch.zeros(num_sample, args.k_prime_classes + args.class_num).cuda()
    feature_bank = torch.zeros(num_sample, args.embed_feat_dim).cuda()

    #AaD
    model.eval()
    all_fea, all_output , _ = inference(model,target_test_dataloader,apply_softmax=True)
    
    all_fea = F.normalize(all_fea)
    feature_bank = all_fea.detach().clone().cuda()
    score_bank  = all_output.detach().clone().cuda()  # .cpu()
    
    for epoch_idx in tqdm(range(args.epochs), ncols=60):
            # Train on target
            loss_dict,score_bank,feature_bank =train(args, model,old_classifier, target_train_dataloader, target_test_dataloader, optimizer,optimizer_c, score_bank,feature_bank , epoch_idx)
            args.logger.info("Epoch: {}/{},          train_all_loss:{:.3f},\n,".format(epoch_idx+1, args.epochs,loss_dict["all_pred_loss"]))

            # Evaluate on target
            hscore, knownacc, unknownacc = test(args, model, target_test_dataloader, src_flg=False)
            args.logger.info("Current: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(hscore, knownacc, unknownacc))

            if hscore >= best_h_score:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc
                best_epoch_idx = epoch_idx

            args.logger.info("Best   : H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(best_h_score, best_known_acc, best_unknown_acc))
        
if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)
    
    args.checkpoint = os.path.join("checkpoints", args.dataset, "source_{}".format(args.s_idx),\
                    "source_{}_{}".format(args.source_train_type, args.target_label_type),
                    "latest_source_checkpoint.pth")
    args.reset = False
    main(args)