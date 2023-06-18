#!/bin/bash

export PYTHONPATH=$PWD/third_party:$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=/conda/lib:$LD_LIBRARY_PATH

ln -s /opt/ml/disk/gnn/hgnn
ln -s /opt/ml/model results
#pip install PyYAML --upgrade -i https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
# --extra-index-url https://mirrors.tencent.com/pypi/simple/
python -u train_graph.py --data_path hgnn \
                  --dataset PROTEINS \
                  --save_dir results \
                  --is_probH \
                  --add_self_loop \
                  ##learnable_adj## \
                  --m_prob ##m_prob## \
                  --epochs 450 \
                  --seed 0 \
                  --K_neigs ##K_neigs## \
                  --type mutigcn \
                  --hidden ##hidden## \
                  --mlp_hidden ##mlp_hidden## \
                  --print_freq 20 \
                  --sampling_percent 1 \
                  --nhiddenlayer 1 \
                  --nbaseblocklayer ##nbaseblocklayer## \
                  --lr 0.01 \
                  --lr_adj ##lr_adj## \
                  --lamda ##lamda## \
                  --theta ##theta## \
                  --dropout ##dropout## \
                  --wd_adj ##wd_adj## \
                  --adj_loss_coef ##adj_loss_coef## \
                  --weight_decay ##weight_decay## \
                  --inputlayer gcn \
                  --outputlayer gcn \
                  --debug \
                  --batch_size ##batch_size## \
                  --iters_per_epoch 1 \
                  --gpu 0|tee log
##pubmed
#python -u train.py --data_path ../data/hgnn/hypergcn \
#                  --on_dataset NTU2012  \
#                  --activate_dataset cocitation/pubmed \
#                  --gvcnn_feature_structure \
#                  --use_gvcnn_feature \
#                  --save_dir ../model/hgnn/pubmed \
#                  --is_probH \
#                  --m_prob 1 \
#                  --add_self_loop \
#                  --epochs 300 \
#                  --seed 1000 \
#                  --K_neigs 11 \
#                  --type mutigcn \
#                  --hidden 64 \
#                  --mlp_hidden 64 \
#                  --print_freq 40 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 1 \
#                  --lr 0.01 \
#                  --lamda 0.0 \
#                  --dropout 0.496314457945 \
#                  --wd_adj 0.534579914794 \
#                  --weight_decay 0.029612134041 \
#                  --gpu 5 \
#                  --theta 0.1 \
#                  --lr_adj 0.0281404388 \
#                  --adj_loss_coef 0.00800880392026 \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --learnable_adj
## hypergcn
#python -u train.py --data_path hgnn/hypergcn \
#                  --on_dataset NTU2012 \
#                  --activate_dataset cocitation/citeseer \
#                  --mvcnn_feature_structure \
#                  --use_mvcnn_feature \
#                  --save_dir results \
#                  --is_probH \
#                  --add_self_loop \
#                  ##learnable_adj## \
#                  --m_prob ##m_prob## \
#                  --debug \
#                  --epochs 500 \
#                  --debug \
#                  --seed 1000 \
#                  --K_neigs ##K_neigs## \
#                  --type mutigcn \
#                  --hidden ##hidden## \
#                  --mlp_hidden ##mlp_hidden## \
#                  --print_freq 5 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer ##nbaseblocklayer## \
#                  --lr 0.01 \
#                  --lr_adj ##lr_adj## \
#                  --lamda ##lamda## \
#                  --theta ##theta## \
#                  --dropout ##dropout## \
#                  --wd_adj ##wd_adj## \
#                  --adj_loss_coef ##adj_loss_coef## \
#                  --weight_decay ##weight_decay## \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --gpu 0|tee log

#python -u train.py --data_path hgnn \
#                  --on_dataset NTU2012 \
#                  --activate_dataset citeseer \
#                  --mvcnn_feature_structure \
#                  --use_mvcnn_feature \
#                  --save_dir results \
#                  --is_probH \
#                  --add_self_loop \
#                  ##learnable_adj## \
#                  --m_prob ##m_prob## \
#                  --debug \
#                  --epochs 100 \
#                  --seed 1000 \
#                  --K_neigs ##K_neigs## \
#                  --type mutigcn \
#                  --hidden ##hidden## \
#                  --mlp_hidden ##mlp_hidden## \
#                  --print_freq 25 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer ##nbaseblocklayer## \
#                  --lr 0.01 \
#                  --lr_adj ##lr_adj## \
#                  --lamda ##lamda## \
#                  --theta ##theta## \
#                  --dropout ##dropout## \
#                  --wd_adj ##wd_adj## \
#                  --adj_loss_coef ##adj_loss_coef## \
#                  --weight_decay ##weight_decay## \
#                  --inputlayer gcn \
#                  --outputlayer gcn \
#                  --gpu 0|tee log
#python -u train.py --data_path hgnn \
#                  --on_dataset NTU2012 \
#                  --activate_dataset mvgnn \
#                  --mvcnn_feature_structure \
#                  --use_mvcnn_feature \
#                  --save_dir results \
#                  --learnable_adj \
#                  --is_probH \
#                  --m_prob 0.9 \
#                  --debug \
#                  --epochs 41 \
#                  --seed 1000 \
#                  --K_neigs 10 \
#                  --type gcnii \
#                  --hidden 64 \
#                  --print_freq 20 \
#                  --sampling_percent 1 \
#                  --nhiddenlayer 1 \
#                  --nbaseblocklayer 2 \
#                  --lr 0.01 \
#                  --lamda 0.5 \
#                  --dropout 0.6 \
#                  --wd_adj 0.01 \
#                  --weight_decay 5e-4 \
#                  --gpu 0|tee log
grep ^test[^ ]*=* log > result_file
grep ^roce*=* log >> result_file
exit 0
