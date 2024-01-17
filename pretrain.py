import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import rdkit
import sys
from tqdm import tqdm
import numpy as np
from gnn_model import GNN,GNN_codebook
from decoder import Model_decoder  

sys.path.append('./util/')

from data_utils import *


torch.multiprocessing.set_sharing_strategy('file_system')

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def group_node_rep(node_rep, batch_size, num_part):
    group = []
    motif_group=[]
    super_group = []
    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        num_motif = num_part[i][1]
        num_all = num_atom + num_motif + 1
        group.append(node_rep[count:count + num_atom])
        motif_group.append(node_rep[count + num_atom:count + num_all-1])
        super_group.append(node_rep[count + num_all -1])
        count += num_all
    return group,motif_group, super_group

def group_node_rep_nosuper(node_rep_nosuper, batch_size, num_part):
    group = []
    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        group.append(node_rep_nosuper[count:count + num_atom])
        count += num_atom
    return group


def train(model_list, loader, optimizer_list, device,args):
    model,model_decoder = model_list

    model.train()
    model_decoder.train()
    if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0, 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_size = len(batch)

        graph_batch = molgraph_to_graph_data(batch).to(device)
   
        node_rep,loss_q = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr,graph_batch.num_part)

        num_part = graph_batch.num_part
        node_rep, motif_rep, super_node_rep = group_node_rep(node_rep, batch_size, num_part)


        loss, bond_if_auc, bond_if_ap, bond_type_acc, atom_type_acc, atom_num_rmse, bond_num_rmse = model_decoder(batch, node_rep, super_node_rep)

        weighted_loss=loss+0.1*loss_q

        optimizer_list.zero_grad()

        weighted_loss.backward()
        optimizer_list.step()

        if_auc += bond_if_auc
        if_ap += bond_if_ap
        type_acc += bond_type_acc
        a_type_acc += atom_type_acc
        a_num_rmse += atom_num_rmse
        b_num_rmse += bond_num_rmse

        if (step+1) % 20 == 0:
            if_auc = if_auc / 20 
            if_ap = if_ap / 20 
            type_acc = type_acc / 20 
            a_type_acc = a_type_acc / 20
            a_num_rmse = a_num_rmse / 20
            b_num_rmse = b_num_rmse / 20

            print('Batch:',step,'loss:',loss.item())
            if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0, 0
           

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='./data/zinc/all.txt',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--output_model_file', type=str, default='./saved_model/pretrain.pth',
                        help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default=12, help='number of workers for dataset loading')
    parser.add_argument("--hidden_size", type=int, default=300, help='hidden size')
    parser.add_argument('--beta', type=float, default=0.1, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = MoleculeDataset(args.dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x:x, drop_last=True)

    model = GNN_codebook(args.num_layer, args.emb_dim, args=args,JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    model_decoder = Model_decoder(args.hidden_size, device).to(device)

    model_list = [model,model_decoder]
    optimizer = optim.Adam([{"params":model.parameters()},{"params":model_decoder.parameters()}], lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print('====epoch',epoch)
        train(model_list, loader, optimizer, device,args)

        if not args.output_model_file == "":
            torch.save(model.state_dict(), args.output_model_file)



if __name__ == "__main__":
    main()
