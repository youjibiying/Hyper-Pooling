#!/bin/bash
#python -u train_graph.py --data_path ../data/hgnn \
#                  --dataset PTC \
#                  --save_dir ../model/hgnn/PTC \
#                  --is_probH \
#                  --m_prob 1.22329227718 \
#                  --add_self_loop \
#                  --epochs 200 \
#                  --seed 0 \
#                  --K_neigs 1 \
#                  --type mutigcn \
#                  --hidden 16 \
#                  --mlp_hidden 64 \
#                  --print_freq 1 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 0 \
#                  --lr 0.01 \
#                  --lamda 0.0 \
#                  --dropout 0.743434497283 \
#                  --wd_adj 0.534579914794 \
#                  --weight_decay 0.0694593086472 \
#                  --gpu 3 \
#                  --theta 0.640962630006 \
#                  --lr_adj 0.0281404388 \
#                  --adj_loss_coef 0.00800880392026 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --iters_per_epoch 1 \
#                  --batch_size 64 \
#                  --debug \
#                  --withbn \
#                  --tensorboard \
#                  --learnable_adj \
##                  --debug \
## gcnii PTC
#python -u train_graph.py --data_path ../data/hgnn \
#                  --dataset PTC \
#                  --save_dir ../model/hgnn/PTC_gcnii \
#                  --is_probH \
#                  --m_prob 1.22329227718 \
#                  --add_self_loop \
#                  --epochs 200 \
#                  --seed 0 \
#                  --K_neigs 1 \
#                  --type gcnii \
#                  --hidden 64 \
#                  --mlp_hidden 128 \
#                  --print_freq 1 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 8 \
#                  --lr 0.01 \
#                  --lamda 0.726597734121 \
#                  --dropout 0.56198243576 \
#                  --wd_adj 0.0005 \
#                  --weight_decay 0.181617614392 \
#                  --gpu 3 \
#                  --theta 0.640962630006 \
#                  --lr_adj 0.0281404388 \
#                  --adj_loss_coef 0.00800880392026 \
#                  --inputlayer linear \
#                  --outputlayer linear \
#                  --iters_per_epoch 1 \
#                  --batch_size 64 \
#                  --debug \
#                  --withbn \
#                  --learnable_adj \
##                  --tensorboard \
###                  --learnable_adj \
###                  --debug

##  (MUTAG)
#python -u train_graph.py --data_path ../data/hgnn \
#                  --dataset MUTAG \
#                  --save_dir ../model/hgnn/MUTAG \
#                  --is_probH \
#                  --m_prob 1.0 \
#                  --add_self_loop \
#                  --epochs 300 \
#                  --seed 0 \
#                  --K_neigs 1 \
#                  --type mutigcn \
#                  --hidden 64 \
#                  --mlp_hidden 64 \
#                  --print_freq 30 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 2 \
#                  --lr 0.01 \
#                  --lamda 0.726597734121 \
#                  --dropout 0.30555 \
#                  --wd_adj 0.137298901256 \
#                  --weight_decay 0.159380351184 \
#                  --gpu 3 \
#                  --theta 0.25203078301 \
#                  --lr_adj 0.0177499013073 \
#                  --adj_loss_coef 0.0 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --iters_per_epoch 1 \
#                  --batch_size 16 \
#                  --attn_adj \
#                  --learnable_adj \
#                  --residue \
#                  --withbn \
#                  --learnable_adj \
##                  --tensorboard \
#                  --debug

## REDDITMULTI5K \ NCI1  IMDBBINARY MUTAG

# IMDBBINARY
#python -u train_graph.py --data_path ../data/hgnn \
#                  --dataset IMDBBINARY \
#                  --save_dir ../model/hgnn/IMDBBINARY \
#                  --is_probH \
#                  --m_prob 1.0 \
#                  --add_self_loop \
#                  --epochs 500 \
#                  --seed 0 \
#                  --K_neigs 1 \
#                  --type mutigcn \
#                  --hidden 64 \
#                  --mlp_hidden 64 \
#                  --print_freq 10 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 8 \
#                  --lr 0.01 \
#                  --lamda 0.726597734121 \
#                  --dropout 0.30555 \
#                  --wd_adj 0.137298901256 \
#                  --weight_decay 0.159380351184 \
#                  --gpu 3 \
#                  --theta 0.05 \
#                  --lr_adj 0.0177499013073 \
#                  --adj_loss_coef 0.0 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --iters_per_epoch 1 \
#                  --batch_size 32 \
#                  --attn_adj \
#                  --residue \
#                  --learnable_adj \
##                  --debug \

## PTC
#python -u train_graph.py --data_path ../data/hgnn \
#                  --dataset PTC \
#                  --save_dir ../model/hgnn/PTC \
#                  --is_probH \
#                  --m_prob 1.0 \
#                  --add_self_loop \
#                  --epochs 300 \
#                  --seed 0 \
#                  --K_neigs 1 \
#                  --type mutigcn \
#                  --hidden 64 \
#                  --mlp_hidden 64 \
#                  --print_freq 10 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 0 \
#                  --lr 0.01 \
#                  --lamda 0.726597734121 \
#                  --dropout 0.265039220971 \
#                  --wd_adj 0.137298901256 \
#                  --weight_decay 0.159380351184 \
#                  --gpu 3 \
#                  --theta 0.1 \
#                  --lr_adj 0.0177499013073 \
#                  --adj_loss_coef 0.0 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --iters_per_epoch 1 \
#                  --batch_size 16 \
#                  --attn_adj \
#                  --learnable_adj \
##                  --tensorboard \
##                  --debug \
#                  --residue \
##                  --learnable_adj
### IMDBMULTI
#python -u train_graph.py --data_path ../data/hgnn \
#                  --dataset IMDBMULTI \
#                  --save_dir ../model/hgnn/IMDBMULTI \
#                  --is_probH \
#                  --m_prob 1.0 \
#                  --add_self_loop \
#                  --epochs 300 \
#                  --seed 0 \
#                  --K_neigs 1 \
#                  --type mutigcn \
#                  --hidden 64 \
#                  --mlp_hidden 64 \
#                  --print_freq 10 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 1 \
#                  --lr 0.01 \
#                  --lamda 0.726597734121 \
#                  --dropout 0.719082438621 \
#                  --wd_adj 0.137298901256 \
#                  --weight_decay 0.0824319607811 \
#                  --gpu 3 \
#                  --theta 0.1 \
#                  --lr_adj 0.0177499013073 \
#                  --adj_loss_coef 0.0 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --iters_per_epoch 1 \
#                  --batch_size 64 \
#                 --learnable_adj \
#                   --residue \
##                  --tensorboard \
##                  --debug \
##                  --learnable_adj
## IMDBBINARY
#python -u train_graph.py --data_path ../data/hgnn \
#                  --dataset IMDBBINARY \
#                  --save_dir ../model/hgnn/IMDBBINARY \
#                  --is_probH \
#                  --m_prob 1.0 \
#                  --add_self_loop \
#                  --epochs 300 \
#                  --seed 0 \
#                  --K_neigs 1 \
#                  --type mutigcn \
#                  --hidden 64 \
#                  --mlp_hidden 64 \
#                  --print_freq 10 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 1 \
#                  --lr 0.01 \
#                  --lamda 0.726597734121 \
#                  --dropout 0.198861007733 \
#                  --wd_adj 0.137298901256 \
#                  --weight_decay 0.0710879234017 \
#                  --gpu 7 \
#                  --theta 0.1 \
#                  --lr_adj 0.0177499013073 \
#                  --adj_loss_coef 0.0 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --iters_per_epoch 1 \
#                  --batch_size 16 \
#                  --attn_adj \
##                   --residue \
###                  --learnable_adj \
##                  --tensorboard \
##                  --debug \
##                  --learnable_adj

## COLLAB
python -u train_graph.py --data_path ../data/hgnn \
                  --dataset COLLAB \
                  --save_dir ../model/hgnn/COLLAB \
                  --is_probH \
                  --m_prob 1.0 \
                  --add_self_loop \
                  --epochs 300 \
                  --seed 0 \
                  --K_neigs 1 \
                  --type mutigcn \
                  --hidden 32 \
                  --mlp_hidden 32 \
                  --print_freq 10 \
                  --sampling_percent 1 \
                  --nhiddenlayer 1 \
                  --nbaseblocklayer 1 \
                  --lr 0.01 \
                  --lamda 0.726597734121 \
                  --dropout 0.36549098171 \
                  --wd_adj 0.137298901256 \
                  --weight_decay 0.0796916037291 \
                  --gpu 6 \
                  --theta 0.1 \
                  --lr_adj 0.0177499013073 \
                  --adj_loss_coef 0.0 \
                  --inputlayer gcn \
                  --outputlayer gcn \
                  --iters_per_epoch 1 \
                  --batch_size 32 \
                  --fold_idx 4 \
                  --learnable_adj \
#                   --residue \
#                  --tensorboard \
#                  --debug \
#                  --learnable_adj

## PROTEINS
#python -u train_graph.py --data_path ../data/hgnn \
#                  --dataset PROTEINS \
#                  --save_dir ../model/hgnn/PROTEINS \
#                  --is_probH \
#                  --m_prob 1.0 \
#                  --add_self_loop \
#                  --epochs 300 \
#                  --seed 0 \
#                  --K_neigs 1 \
#                  --type mutigcn \
#                  --hidden 64 \
#                  --mlp_hidden 64 \
#                  --print_freq 20 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 1 \
#                  --lr 0.01 \
#                  --lamda 0.726597734121 \
#                  --dropout 0.113235564715 \
#                  --wd_adj 0.137298901256 \
#                  --weight_decay 0.0755587618139 \
#                  --gpu 7 \
#                  --theta 0.1 \
#                  --lr_adj 0.0177499013073 \
#                  --adj_loss_coef 0.0 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --iters_per_epoch 1 \
#                  --batch_size 16 \
###                  --attn_adj \
##                  --learnable_adj \
##                   --residue \
###                  --learnable_adj \

# NCI1
#python -u train_graph.py --data_path ../data/hgnn \
#                  --dataset NCI1 \
#                  --save_dir ../model/hgnn/NCI1 \
#                  --is_probH \
#                  --m_prob 1.0 \
#                  --add_self_loop \
#                  --epochs 300 \
#                  --seed 0 \
#                  --K_neigs 1 \
#                  --type mutigcn \
#                  --hidden 64 \
#                  --mlp_hidden 64 \
#                  --print_freq 10 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 1 \
#                  --lr 0.01 \
#                  --lamda 0.726597734121 \
#                  --dropout 0.631073384651 \
#                  --wd_adj 0.137298901256 \
#                  --weight_decay 0.0284029143502 \
#                  --gpu 2 \
#                  --theta 0.1 \
#                  --lr_adj 0.0177499013073 \
#                  --adj_loss_coef 0.0 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --iters_per_epoch 1 \
#                  --batch_size 16 \
#                  --learnable_adj \
#                  --attn_adj \
#                   --residue \
##                  --learnable_adj \
