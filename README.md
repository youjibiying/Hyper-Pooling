# HyperGNN
# dataset
## protein
    - gpcr
    avg_nodes=375.72331154684093+-112.36729632005856
    avg_edges=579.6644880174292+-183.37287946123774
    avg_edges_degree=5.2264692249366895+-0.10795070719133516
    len(seqlist):len(set(seqlist))=456:376

    - kinase
    avg_nodes=294.5273159144893+-84.94280506729615
    avg_edges=477.32304038004753+-142.0712415926186
    avg_edges_degree=5.171643721885741+-0.07001297175794546
    len(seqlist):len(set(seqlist))=421:335
    
    data format [row, col ,value] is coo sparse matrix
 
# model
topkpooling 
Task: graph classification
 
# training
python train_graph.py --data_path ../data/hgnn --dataset PROTEINS --save_dir ../model/hgnn/PROTEINS --is_probH --m_prob 1.0 --add_self_loop --epochs 3 --seed 0 --K_neigs 1 --type topkpooling --hidden 64 --mlp_hidden 64 --print_freq 20 --sampling_percent 1 --nhiddenlayer 1 --nbaseblocklayer 1 --lr 0.01 --lamda 0.726597734121 --dropout 0.113235564715 --wd_adj 0.137298901256 --weight_decay 0.0755587618139 --gpu 7 --theta 0.1 --lr_adj 0.0177499013073 --adj_loss_coef 0.0 --inputlayer gcn --outputlayer dense --iters_per_epoch 1 --batch_size 16 --debug

接下来
1. 将 G换成 H，然后topk 只对H作处理
2. scores 计算部分换成HGCN
3. 学习AAAI的ASAP算法，整合到当前模型里边
