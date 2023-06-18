import argparse
import os
import os.path as osp



def parsering():
    # Training settings
    parser = argparse.ArgumentParser()
    # Training parameter
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Disable validation during training.')
    parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
    parser.add_argument('--print_freq', type=int, default=20, help='log frequency.')
    parser.add_argument('--epochs', type=int, default=800,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,help='Initial learning rate.')
    parser.add_argument('--lr_adj', type=float, default=0.001,help='Initial learning rate.')
    parser.add_argument('--lradjust', action='store_true',
                        default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--mixmode", action="store_true",
                        default=False, help="Enable CPU GPU mixing mode.")
    parser.add_argument("--warm_start", default="",
                        help="The model name to be loaded for warm start.")
    parser.add_argument('--debug', action='store_true',
                        default=False, help="Enable the detialed training output.")
    parser.add_argument('--dataset', default="Proteins", help="The data set")
    parser.add_argument('--data_path', default="../data/hgnn", help="The data path.")
    parser.add_argument("--early_stopping", type=int,
                        default=100,
                        help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
    parser.add_argument("--tensorboard", action='store_true', default=False,
                        help="Disable writing logs to tensorboard")
    # add save_dir
    parser.add_argument('--save_dir', type=str, default="../model/hgnn/NTU_mvcnn", help="The data path.")

    # Model parameter
    parser.add_argument('--type', default='gcnii',
                        help="Choose the model to be trained.(mutigcn, resgcn, densegcn, inceptiongcn)")
    parser.add_argument('--inputlayer', default='linear',
                        help="The input layer of the model.")
    parser.add_argument('--outputlayer', default='linear',
                        help="The output layer of the model.")
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--withbn', action='store_true', default=False,
                        help='Enable Bath Norm GCN')
    parser.add_argument('--residue', action='store_true', default=False,
                        help='Enable Bath Norm GCN')
    parser.add_argument('--withloop', action="store_true", default=False,
                        help="Enable loop layer GCN")
    parser.add_argument('--nhiddenlayer', type=int, default=1,
                        help='The number of hidden layers.')
    parser.add_argument("--normalization", default="AugNormAdj",
                        help="The normalization on the adj matrix.")
    parser.add_argument("--sampling_percent", type=float, default=1.0,
                        help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
    parser.add_argument("--pretrain_sampling_percent", type=float, default=1.0,
                        help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
    # parser.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, mutigcn, inceptiongcn).")
    parser.add_argument("--nbaseblocklayer", type=int, default=1,
                        help="The number of layers in each baseblock")  # same as '--layer' of gcnii
    parser.add_argument("--aggrmethod", default="nores",
                        help="The aggrmethod for the layer aggreation. The options includes add and concat."
                             " Only valid in resgcn, densegcn and inecptiongcn")
    parser.add_argument("--task_type", default="full",
                        help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

    # argument for gcnii
    parser.add_argument('--wd_adj', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
    # parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--gpu', type=int, default=0, help='device id')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
    parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')  # gcnii*

    # argument for learnable adj
    parser.add_argument('--learnable_adj', action='store_true', default=False, help='learnable adj')
    parser.add_argument('--theta', type=float, default=0.2, help='adj=theta * add_edge_adj + (1 - theta) * core_edge_adj')
    parser.add_argument('--attn_adj', action='store_true', default=False, help='learnable adj by self attention')
    parser.add_argument('--add_self_loop', action='store_true', default=False, help='Don"t add self loop in adj')  # gcnii*
    parser.add_argument('--mlp_hidden', type=int, default=64,
                        help='Number of hidden units.')

    # argument for HGCN
    parser.add_argument('--activate_dataset', type=str, default="cora", help="The gcn common dataset.")
    parser.add_argument('--K_neigs', type=int, default=10, help="the k of knn")
    parser.add_argument('--is_probH', action='store_true', default=False, help='Don"t using probability distance map')
    parser.add_argument('--m_prob', type=float, default=1.0, help="row normalization of H or dist D=D/m_prob")
    parser.add_argument('--gamma', type=float, default=0.9, help="")
    parser.add_argument('--adj_loss_coef', type=float, default=0.1, help="")
    parser.add_argument('--milestones', type=list, default=[100], help="")
    parser.add_argument('--on_dataset', type=str, default="NTU2012",
                        help="elect the dataset you use, ModelNet40 or NTU2012.", choices=['ModelNet40','NTU2012'])
    parser.add_argument('--mvcnn_feature_structure', action='store_true', default=False,
                        help='use_mvcnn_feature_for_structure')
    parser.add_argument('--gvcnn_feature_structure', action='store_true', default=False,
                        help='use_gvcnn_feature_for_structure')
    parser.add_argument('--use_gvcnn_feature', action='store_true', default=False,
                        help='use_gvcnn_feature_add to features X')
    parser.add_argument('--use_mvcnn_feature', action='store_true', default=False,
                        help='use_gvcnn_feature_add  to features X')

    # powerful-gnns(GIN) graph level

    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    # 後面可以刪除
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')

    # graument for hypergcn ( hyper graph dataset
    parser.add_argument('--split', type=int, default=1, help='train-test split used for the dataset')
    # argument for pooling
    parser.add_argument('--ks', nargs='+', type=float, default=[0.9, 0.8, 0.7],help='sample radio each at layer')

    args = parser.parse_args()

    return args

def check_dir(folder, mk_dir=True):
    if not osp.exists(folder):
        if mk_dir:
            print(f'making direction {folder}!')
            os.mkdir(folder)
        else:
            raise Exception(f'Not exist direction {folder}')


def check_dirs(args):
    check_dir(args.data_path, mk_dir=False)

    check_dir(args.save_dir)

def train_args():
    args = parsering()
    # args.modelnet40_ft = os.path.join(args.data_path, 'ModelNet40_mvcnn_gvcnn.mat')
    # args.ntu2012_ft = os.path.join(args.data_path, 'NTU2012_mvcnn_gvcnn.mat')
    args.citation_root = os.path.join(args.data_path, 'dhgnn')
    # check_dirs(args)
    return args
