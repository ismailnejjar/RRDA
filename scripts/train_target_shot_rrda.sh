gpuid=${1:-0}
random_seed=${2:-2021}

export CUDA_VISIBLE_DEVICES=$gpuid

echo "OSDA Adaptation ON Office"
python train_target_rrda_shot.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001   --target_label_type OSDA
python train_target_rrda_shot.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001   --target_label_type OSDA
python train_target_rrda_shot.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001   --target_label_type OSDA
python train_target_rrda_shot.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001   --target_label_type OSDA
python train_target_rrda_shot.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001   --target_label_type OSDA
python train_target_rrda_shot.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001   --target_label_type OSDA

echo "OSDA Adaptation ON VisDA"
python train_target_rrda_shot.py --backbone_arch resnet50 --lr 0.0001 --dataset VisDA  --target_label_type OSDA --lam_pseudo 0.4

echo "OSDA Adaptation ON Office-Home"
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 0 --t_idx 1 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 0 --t_idx 2 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 0 --t_idx 3 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 1 --t_idx 0 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 1 --t_idx 2 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 1 --t_idx 3 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 2 --t_idx 0 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 2 --t_idx 1 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 2 --t_idx 3 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 3 --t_idx 0 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 3 --t_idx 1 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1
python train_target_rrda_shot.py --dataset OfficeHome --s_idx 3 --t_idx 2 --lr 0.001   --target_label_type OSDA --lam_pseudo 0.1