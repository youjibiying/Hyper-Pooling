import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import numpy as np
import copy
import random
from tqdm import tqdm
import time, os

from datasets.util import separate_data
from models.graphcnn import GraphCNN
from parsing import train_args
from dataset.data_load import load_data

criterion = nn.CrossEntropyLoss()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, device, train_graphs, test_graphs, optimizer, epoch, scheduler):
    model.train()

    total_iters = args.iters_per_epoch
    # if args.debug:
    #     pbar = tqdm(range(total_iters), unit='batch')  # range(total_iters)
    # else:
    pbar = range(total_iters)

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = criterion(output, labels)
        if args.debug and torch.isnan(loss):
            print(output)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            scheduler.step(loss)

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        # if args.debug:
        #     pbar.set_description('epoch: %d' % (epoch))
    average_loss = loss_accum / total_iters

    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    # print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test


def main(args):
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    setup_seed(args.seed)
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if args.tensorboard:
        if args.learnable_adj:
            adj_style = 'learnable'
        else:
            adj_style = 'fixed'
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(
            os.path.join(args.save_dir, f'{args.nbaseblocklayer}-layers_{args.type}-{adj_style}-{time_start}'))
    since = time.time()

    device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")

    graphs, num_classes, fea_dim = load_data(data_path=args.data_path, dataset=args.dataset)
    # graphs, num_classes = load_data(args.dataset, args.degree_as_tag, args)

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx, dataset=args.dataset)  # todo 加入val-data

    model = GraphCNN(args.num_layers, args.num_mlp_layers, fea_dim, args.hidden,
                     num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                     args.neighbor_pooling_type, device, args=args).to(device)
    print(model.hgcn)
    #
    # p1, p2 = [], []
    # for n, p in model.named_parameters():
    #     # print(n, p.shape)
    #     if '.dynamic_adj.' in n:
    #         p2.append(p)
    #     else:
    #         p1.append(p)
    # optimizer = optim.Adam([
    #     {'params': p1, 'weight_decay': args.weight_decay, 'lr': args.lr},
    #     {'params': p2, 'weight_decay': args.wd_adj, 'lr': args.lr_adj}])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=100,
                                                           verbose=False,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                           eps=1e-08)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, best_epoch, best_val_loss = -1, 0, -1.0
    loss_pred, loss_adj = 0, 0
    for epoch in range(1, args.epochs + 1):
        # if epoch==55 and args.debug:
        #     print(model.hgcn.state_dict())
        avg_loss = train(args, model, device, train_graphs, test_graphs, optimizer, epoch,
                         scheduler)
        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)

        if acc_test > best_acc:
            early_stop = 0
            best_acc = acc_test
            best_epoch = epoch
            best_val_loss = avg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            early_stop += 1
            if early_stop > args.early_stopping:
                print('early stop at epoch', epoch)
                break

        # acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)
        if epoch % args.print_freq == 0:
            print(
                f'epoch: {epoch} avgLoss: {avg_loss:.4f} pre_loss: {loss_pred:.4f} adj_loss: {loss_adj:.4f} '
                f'lr: {optimizer.param_groups[0]["lr"]} train_acc: {acc_train:.4f} test_acc: {acc_test:.4f}')
            print(f'Best val Acc: {best_acc:4f} at epoch: {best_epoch}')
        if args.tensorboard:
            writer.add_scalar('loss/train', avg_loss, epoch)
            writer.add_scalar('acc/train', acc_train, epoch)
            writer.add_scalar('acc/val', acc_test, epoch)
            writer.add_scalar('best_acc/val', best_acc, epoch)
        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")

        test_acc = best_acc
    print(f"test_acc={test_acc}\n"
          f"best_val_acc={best_acc}\n"
          f"best_val_epoch={best_epoch}\n"
          f"best_val_loss={best_val_loss}")
    return test_acc


if __name__ == '__main__':
    args = train_args()
    print(args)
    if args.debug:
        fold_nums = [args.fold_idx]  # 0
    else:
        fold_nums = range(args.fold_idx, 10)  #
    results = []
    for fold_idx in fold_nums:
        args.fold_idx = fold_idx
        print(f"fold:{args.fold_idx}/{fold_nums}")
        results.append(main(args))
        print(results)
    print(args)
    results = np.array(results)
    print(f"avg_test_acc={results.mean()} \n"
          f"std={results.std()}")
