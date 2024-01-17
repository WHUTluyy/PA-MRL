import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, remove_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
from set_transformer import SetTransformer
import sys
sys.path.append("..") 
from codebook import Codebook

num_atom_type = 121  # including the extra motif tokens and graph token
num_chirality_tag = 11  # degree

num_bond_type = 7
num_bond_direction = 3

def group_node_rep(node_rep, batch_size, num_part):
    motif_group=[]
    count = 0
    for i in range(batch_size):
        num_atom = int(num_part[i][0].item())
        num_motif = int(num_part[i][1].item())
        num_all = num_atom + num_motif + 1
        motif_group.append(node_rep[count + num_atom:count + num_all-1])
        count += num_all
    return motif_group

def replace_node_rep(node_rep,motif_group, batch_size, num_part):
    count = 0
    for i in range(batch_size):
        num_atom = int(num_part[i][0].item())
        num_motif = int(num_part[i][1].item())
        num_all = num_atom + num_motif + 1
        node_rep[count + num_atom:count + num_all-1] = motif_group[i][:]
        count += num_all
    return node_rep

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(
            emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(
            edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(
            edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        # assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(
            edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self,
                 emb_dim,
                 heads=2,
                 negative_slope=0.2,
                 dropout=0.,
                 bias=True):
        # "Add" aggregation.
        super(GATConv, self).__init__(node_dim=0, aggr='add')

        self.in_channels = emb_dim
        self.out_channels = emb_dim
        self.edge_dim = emb_dim  # new
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.weight = torch.nn.Parameter(torch.Tensor(
            emb_dim, heads * emb_dim))    # emb(in) x [H*emb(out)]
        # 1 x H x [2*emb(out)+edge_dim]    # new
        self.att = torch.nn.Parameter(torch.Tensor(
            1, heads, 2 * emb_dim + self.edge_dim))
        self.edge_update = torch.nn.Parameter(torch.Tensor(
            emb_dim + self.edge_dim, emb_dim))   # [emb(out)+edge_dim] x emb(out)  # new

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.edge_update)  # new
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        edge_attr = self.edge_embedding1(
            edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)

        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_edges = torch.zeros(
            x.size(0), edge_attr.size(1)).to(edge_index.device)
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_i, x_j, size_i, edge_index_i, edge_attr):

        edge_attr = edge_attr.unsqueeze(1).repeat(
            1, self.heads, 1)  # (E+N) x H x edge_dim  # new
        # (E+N) x H x (emb(out)+edge_dim)   # new
        x_j = torch.cat([x_j, edge_attr], dim=-1)
        # (E+N) x H x emb(out)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        alpha = (torch.cat([x_i, x_j], dim=-1) *
                 self.att).sum(dim=-1)  # (E+N) x H

        alpha = F.leaky_relu(alpha, self.negative_slope)
        # Computes a sparsely evaluated softmax
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # (E+N) x H x (emb(out)+edge_dim)
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = torch.mm(aggr_out, self.edge_update)  # N x emb(out)  # new

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        edge_index = edge_index[0]
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(
            edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]

        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.elu(h), self.drop_ratio,
                              training=self.training)  # relu->elu

            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]

        return node_representation

class GNN_codebook(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim,args ,JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN_codebook, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.codebook=Codebook(args=args)
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        # ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, num_part= argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])


        h_list = [x]
        loss_q=0
        for layer in range(self.num_layer):
            if layer == self.num_layer - 1:
                nodes_rep=h_list[-1]
                motif_group = group_node_rep(nodes_rep, len(num_part), num_part)
                motif_group_q, loss_q,index_list=self.codebook(motif_group)
                # -------------visual-------------------
                # file_path = 'tensor_dic.pt'
                # loaded_dic={}
                # # with open(file_path, 'rb') as file:
                # #     loaded_dic = torch.load(file)
                # for i in range(len(index_list)):
                #     integers = index_list[i].tolist()  # 将张量转换为 Python 列表
                #     for j in range(len(integers)):
                #         key=integers[j]
                #         if key in loaded_dic:
                #             loaded_dic[key].append(motif_group[i][j])
                #         else:
                #             loaded_dic[key]=[motif_group[i][j]]
                # with open(file_path, 'wb') as file:
                #     torch.save(loaded_dic, file)
                # -------------visual-------------------
                nodes_rep=replace_node_rep(nodes_rep,motif_group_q,len(num_part),num_part)
                h_list[-1]=nodes_rep

            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation,loss_q

class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        gnn_type: gin, gcn, graphsage, gat

    """

    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(
                (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim*2, self.emb_dim),
                torch.nn.ELU(),
                torch.nn.Linear(self.emb_dim, (self.emb_dim)//2),
                torch.nn.ELU(),
                torch.nn.Linear((self.emb_dim)//2, self.num_tasks))

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def super_node_rep(self, node_rep, batch):
        super_group = []
        for i in range(batch.num_graphs):
            if i != (batch.num_graphs-1) and batch[i] != batch[i+1]:
                super_group.append(node_rep[i, :])
            elif i == (batch.num_graphs - 1):
                super_group.append(node_rep[i, :])
        super_rep = torch.stack(super_group, dim=0)
        return super_rep  # [batch,hiddim]

    def node_rep_(self, node_rep, batch):
        super_group = []
        batch_new = []
        for i in range(batch.num_graphs):
            if i != (batch.num_graphs-1) and batch[i] == batch[i+1]:
                super_group.append(node_rep[i, :])
                batch_new.append(batch[i].item())
        super_rep = torch.stack(super_group, dim=0)
        batch_new = torch.tensor(np.array(batch_new)).to(
            batch.device).to(batch.dtype)
        return super_rep, batch_new

    def mean_pool_(self, node_rep, batch):
        super_group = [[] for i in range(32)]
        for i in range(batch.num_graphs):
            super_group[batch[i]].append(node_rep[i, :])
        node_rep = [torch.stack(list, dim=0).mean(dim=0)
                    for list in super_group]
        super_rep = torch.stack(node_rep, dim=0)
        return super_rep  # [batch,hiddim]

    def node_group_(self, node_rep, batch):
        super_group = [[] for i in range(32)]
        for i in range(batch.num_graphs):
            super_group[batch[i]].append(node_rep[i, :])
        node_rep = [torch.stack(list, dim=0) for list in super_group]
        return node_rep  # [batch,nodenum,hiddim]

    def graph_emb(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        super_rep = self.super_node_rep(node_representation, batch)
        return super_rep

    def all_node_rep_padding(self, node_rep, num_part):
        molecule_group = []
        motif_group = []
        super_rep = []
        count = 0
        batch_size = num_part.shape[0]
        for i in range(batch_size):
            num_atom = int(num_part[i][0])
            num_motif = int(num_part[i][1])
            num_all = num_atom + num_motif + 1
            # -------------visual-------------------
            # file_path = 'tensor_list_baseline.pt'
            # loaded_list=[]
            # with open(file_path, 'rb') as file:
            #     loaded_list = torch.load(file)
            # for j in range(count + num_atom,count + num_all - 1):
            #     loaded_list.append(node_rep[j])
            # with open(file_path, 'wb') as file:
            #     torch.save(loaded_list, file)
            # -------------visual-------------------
            molecule_group.append(node_rep[count:count + num_atom])
            motif_group.append(node_rep[count + num_atom:count + num_all - 1])
            if len(motif_group[-1])==0:
                motif_group[-1]=torch.zeros(1, 512).to(node_rep.device)
            super_rep.append(node_rep[count + num_all - 1])
            count += num_all
        molecule_group,input_molecule_length=pad_tensor_list(molecule_group,node_rep.device)
        # molecule_group = [list.mean(dim=0) for list in molecule_group]
        molecule_rep_padding = torch.stack(molecule_group, dim=0)
        # motif_group = [list.mean(dim=0) for list in motif_group]
        motif_group,input_motif_length = pad_tensor_list(motif_group,node_rep.device)
        motif_rep_padding = torch.stack(motif_group, dim=0)
        super_rep = torch.stack(super_rep, dim=0)
        return molecule_rep_padding, motif_rep_padding, super_rep,input_molecule_length,input_motif_length

    def all_node_rep(self, node_rep, num_part):
        molecule_group = []
        motif_group = []
        count = 0
        batch_size = num_part.shape[0]
        for i in range(batch_size):
            num_atom = int(num_part[i][0])
            num_motif = int(num_part[i][1])
            num_all = num_atom + num_motif + 1
            molecule_group.append(node_rep[count:count + num_atom])
            motif_group.append(node_rep[count + num_atom:count + num_all - 1])
            if len(motif_group[-1])==0:
                motif_group[-1]=torch.zeros(1, 300).to(node_rep.device)
            count += num_all
        molecule_group = [list.mean(dim=0) for list in molecule_group]
        molecule_rep = torch.stack(molecule_group, dim=0)
        motif_group = [list.mean(dim=0) for list in motif_group]
        motif_rep = torch.stack(motif_group, dim=0)
        return molecule_rep, motif_rep
    
    def forward(self, *argv):
        if len(argv) == 5:
            x, edge_index, edge_attr, batch, num_part = argv[0], argv[1], argv[2], argv[3], argv[4]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        # super_rep1=self.super_node_rep(node_representation,batch)

        molecule_rep_padding, motif_rep_padding, super_rep,input_molecule_length,input_motif_length= self.all_node_rep_padding(node_representation, num_part)
        molecule_rep, motif_rep=self.all_node_rep(node_representation,num_part)

        # molecule_rep_padding=self.set_transformer_molecule(molecule_rep_padding)
        # motif_rep_padding=self.set_transformer_motif(motif_rep_padding)
        # molecule_rep_padding=torch.mean(molecule_rep_padding,dim=1)
        # motif_rep_padding=torch.mean(motif_rep_padding,dim=1)
        
        final_rep=torch.cat((molecule_rep, motif_rep), dim=1)

        # molecule_motify,_=self.attention_readout1(molecule_rep,motif_rep,motif_rep)
        # final_rep,_=self.attention_readout2(molecule_motify,super_rep,super_rep)
        
        return self.graph_pred_linear(final_rep)

def pad_tensor_list(tensor_list,device):
    max_length = max([tensor.size(0) for tensor in tensor_list])  # 找到最长的 Tensor 的长度
    input_length=[]
    padded_tensor_list = []
    for tensor in tensor_list:
        padding_length = max_length - tensor.size(0)  # 计算需要补零的长度
        input_length.append(tensor.size(0))
        padded_tensor = torch.cat([tensor, torch.zeros(padding_length, *tensor.shape[1:]).to(device)], dim=0)  # 在末尾补零
        padded_tensor_list.append(padded_tensor)
    return padded_tensor_list,input_length



class GNN_graphpred_codebook(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        gnn_type: gin, gcn, graphsage, gat

    """

    def __init__(self, num_layer, emb_dim, num_tasks,args, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN_graphpred_codebook, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN_codebook(num_layer, emb_dim,args, JK, drop_ratio, gnn_type=gnn_type)

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(
                (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim*3, self.emb_dim),
                torch.nn.ELU(),
                torch.nn.Linear(self.emb_dim, (self.emb_dim)//2),
                torch.nn.ELU(),
                torch.nn.Linear((self.emb_dim)//2, self.num_tasks))

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))
    
    def from_codebook(self,model_file):
        self.codebook.load_state_dict(torch.load(model_file))

    def super_node_rep(self, node_rep, batch):
        super_group = []
        for i in range(batch.num_graphs):
            if i != (batch.num_graphs-1) and batch[i] != batch[i+1]:
                super_group.append(node_rep[i, :])
            elif i == (batch.num_graphs - 1):
                super_group.append(node_rep[i, :])
        super_rep = torch.stack(super_group, dim=0)
        return super_rep  # [batch,hiddim]

    def node_rep_(self, node_rep, batch):
        super_group = []
        batch_new = []
        for i in range(batch.num_graphs):
            if i != (batch.num_graphs-1) and batch[i] == batch[i+1]:
                super_group.append(node_rep[i, :])
                batch_new.append(batch[i].item())
        super_rep = torch.stack(super_group, dim=0)
        batch_new = torch.tensor(np.array(batch_new)).to(
            batch.device).to(batch.dtype)
        return super_rep, batch_new

    def mean_pool_(self, node_rep, batch):
        super_group = [[] for i in range(32)]
        for i in range(batch.num_graphs):
            super_group[batch[i]].append(node_rep[i, :])
        node_rep = [torch.stack(list, dim=0).mean(dim=0)
                    for list in super_group]
        super_rep = torch.stack(node_rep, dim=0)
        return super_rep  # [batch,hiddim]

    def node_group_(self, node_rep, batch):
        super_group = [[] for i in range(32)]
        for i in range(batch.num_graphs):
            super_group[batch[i]].append(node_rep[i, :])
        node_rep = [torch.stack(list, dim=0) for list in super_group]
        return node_rep  # [batch,nodenum,hiddim]

    def graph_emb(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        super_rep = self.super_node_rep(node_representation, batch)
        return super_rep

    def all_node_rep_padding(self, node_rep, num_part):
        molecule_group = []
        motif_group = []
        super_rep = []
        count = 0
        batch_size = num_part.shape[0]
        for i in range(batch_size):
            num_atom = int(num_part[i][0])
            num_motif = int(num_part[i][1])
            num_all = num_atom + num_motif + 1
            molecule_group.append(node_rep[count:count + num_atom])
            # -------------visual-------------------
            # file_path = 'tensor_list.pt'
            # loaded_list=[]
            # with open(file_path, 'rb') as file:
            #     loaded_list = torch.load(file)
            # for j in range(count + num_atom,count + num_all - 1):
            #     loaded_list.append(node_rep[j])
            # with open(file_path, 'wb') as file:
            #     torch.save(loaded_list, file)
            # -------------visual-------------------
            motif_group.append(node_rep[count + num_atom:count + num_all - 1])
            if len(motif_group[-1])==0:
                motif_group[-1]=torch.zeros(1, 512).to(node_rep.device)
            super_rep.append(node_rep[count + num_all - 1])
            count += num_all
        molecule_group,input_molecule_length=pad_tensor_list(molecule_group,node_rep.device)
        molecule_rep_padding = torch.stack(molecule_group, dim=0)
        motif_group,input_motif_length = pad_tensor_list(motif_group,node_rep.device)
        motif_rep_padding = torch.stack(motif_group, dim=0)
        super_rep = torch.stack(super_rep, dim=0)
        return molecule_rep_padding, motif_rep_padding, super_rep,input_molecule_length,input_motif_length

    def all_node_rep(self, node_rep, num_part):
        molecule_group = []
        motif_group = []
        count = 0
        batch_size = num_part.shape[0]
        for i in range(batch_size):
            num_atom = int(num_part[i][0])
            num_motif = int(num_part[i][1])
            num_all = num_atom + num_motif + 1
            molecule_group.append(node_rep[count:count + num_atom])
            motif_group.append(node_rep[count + num_atom:count + num_all - 1])
            if len(motif_group[-1])==0:
                motif_group[-1]=torch.zeros(1, 512).to(node_rep.device)
            count += num_all
        # motif_group_Q,_ = self.codebook(motif_group)
        # motif_group_Q = [list.mean(dim=0) for list in motif_group_Q]
        # motif_rep_Q = torch.stack(motif_group_Q, dim=0)
        molecule_group = [list.mean(dim=0) for list in molecule_group]
        molecule_rep = torch.stack(molecule_group, dim=0)
        motif_group = [list.mean(dim=0) for list in motif_group]
        motif_rep = torch.stack(motif_group, dim=0)
        return molecule_rep, motif_rep
    
    def forward(self, *argv):
        if len(argv) == 5:
            x, edge_index, edge_attr, batch, num_part = argv[0], argv[1], argv[2], argv[3], argv[4]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation,loss_q = self.gnn(x, edge_index, edge_attr,num_part)

        molecule_rep_padding, motif_rep_padding, super_rep,input_molecule_length,input_motif_length= self.all_node_rep_padding(node_representation, num_part)
        molecule_rep, motif_rep=self.all_node_rep(node_representation,num_part)
        
        final_rep=torch.cat((molecule_rep,motif_rep, super_rep), dim=1)
        
        return self.graph_pred_linear(final_rep)

if __name__ == "__main__":
    pass