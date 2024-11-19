import os
import faiss
import torch 
import shutil 
import numpy as np

from tqdm import tqdm 
from model.SFUniDA import SFUniDA 
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader

from config.model_config import build_args
from utils.net_utils import set_logger, set_random_seed
from utils.net_utils import compute_h_score, Entropy

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
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



def train(args, model, train_dataloader, test_dataloader, optimizer,feature_bank,score_bank, epoch_idx=0.0):
    
    model.eval()
    
    mem_label, ENT_THRESHOLD = obtain_label(args,model,test_dataloader)
    model.train()

    all_pred_loss_stack = []

    iter_idx = epoch_idx * len(train_dataloader)
    iter_max = args.epochs * len(train_dataloader)
    for imgs_train, _, imgs_label, imgs_idx in tqdm(train_dataloader, ncols=60):
        
        iter_idx += 1
        imgs_train = imgs_train.cuda()

        pred = mem_label[imgs_idx]
        features_test , outputs_test  = model(imgs_train, apply_softmax=False)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        
        outputs_test_known = outputs_test[pred < args.class_num, :]

        features_known = features_test[pred < args.class_num, :]
        known_inx = imgs_idx[pred < args.class_num]

        pred = pred[pred < args.class_num]

        if len(pred) == 0:
            print(tt)
            del features_test
            del outputs_test
            tt += 1
            continue
            
        alpha = (1 + 10 * iter_idx / iter_max)**(-1)
        
        classifier_loss = 0
        with torch.no_grad():
            output_f_norm = F.normalize(features_known)
            output_f_ = output_f_norm.detach().clone()

            softmax_out_known = nn.Softmax(dim=1)(outputs_test_known)
            pred_bs = softmax_out_known

            feature_bank[known_inx] = output_f_.detach().clone()
            score_bank[known_inx] = pred_bs.detach().clone()

            distance = output_f_ @ feature_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=3 + 1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]  #batch x K x C

            fea_near = feature_bank[idx_near]  #batch x K x num_dim
        # nn
        softmax_out_un = softmax_out_known.unsqueeze(1).expand(-1, 3, -1)  # batch x K x C

        loss = torch.mean((F.kl_div(softmax_out_un,
                                    score_near,
                                    reduction='none').sum(-1)).sum(1))  #

        mask = torch.ones((imgs_train.shape[0], imgs_train.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = softmax_out.T

        dot_neg = softmax_out @ copy  # batch x batch
        dot_neg = (dot_neg * mask.cuda()).sum(-1)  #batch
        neg_pred = torch.mean(dot_neg)
        loss += neg_pred * alpha
        
        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_pred_loss_stack.append(loss.cpu().item())

    train_loss_dict = {}
    train_loss_dict["all_pred_loss"] = np.mean(all_pred_loss_stack)

    return train_loss_dict,feature_bank,score_bank

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

    h_score, known_acc, unknown_acc, _ = compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flg, open_thresh=args.w_0)
    return h_score, known_acc, unknown_acc
    
def obtain_label(args, model, dataloader):
    with torch.no_grad():
        start_test = True
        for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
            imgs_test = imgs_test.cuda() 
            feas, outputs = model(imgs_test, apply_softmax=False)
    
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = imgs_label.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, imgs_label.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float().cpu() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    ent = torch.sum(-all_output * torch.log(all_output + 1e-6), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()

    
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    labels = kmeans.predict(ent.reshape(-1,1))

    idx = np.where(labels==1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0]

    all_fea = all_fea[known_idx,:]
    all_output = all_output[known_idx,:]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]
    ENT_THRESHOLD = (kmeans.cluster_centers_).mean()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>0)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    guess_label = args.class_num * np.ones(len(all_label), )
    guess_label[known_idx] = pred_label

    acc = np.sum(guess_label == all_label.float().numpy()) / len(all_label_idx)
    log_str = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD, accuracy*100, acc*100)

    return guess_label.astype('int'), ENT_THRESHOLD


def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    
    model = SFUniDA(args)
    
    model = model.cuda()


    save_dir = os.path.join(this_dir, "checkpoints_aad", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type,"{}".format(args.source_train_type))
    
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
    
    param_group = []
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]
    
    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    
    for k, v in model.class_layer.named_parameters():
        v.requires_grad = False  
        
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
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

    best_h_score = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0
    best_epoch_idx = 0
    
    num_sample = len(target_test_dataloader.dataset)
    
    score_bank = torch.zeros(num_sample, args.class_num).cuda()
    feature_bank = torch.zeros(num_sample, args.embed_feat_dim).cuda()

    #AaD
    model.eval()
    all_fea, all_output , _ = inference(model,target_test_dataloader,apply_softmax=True)
    
    all_fea = F.normalize(all_fea)
    feature_bank = all_fea.detach().clone().cuda()
    score_bank  = all_output.detach().clone().cuda()  # .cpu()
    
    for epoch_idx in tqdm(range(args.epochs), ncols=60):
        # Train on target
        loss_dict,feature_bank,score_bank =train(args, model, target_train_dataloader, target_test_dataloader, optimizer,feature_bank,score_bank , epoch_idx)
        args.logger.info("Epoch: {}/{}, train_all_loss:{:.3f},\n".format(epoch_idx+1, args.epochs,loss_dict["all_pred_loss"], ))
        
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