import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.emb_dim
        self.beta = args.beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

        self.empty_tensor = torch.empty(0)

    def forward(self, z):
        # z = z.permute(0, 2, 3, 1).contiguous()
        index_list=[]
        z_q=[]
        loss=0
        for z_t in z:
            if torch.numel(z_t)==0:
                z_q.append(z_t)
                index_list.append(self.empty_tensor)
                continue
            z_flattened = z_t.view(-1, self.latent_dim)

            d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - \
                2*(torch.matmul(z_flattened, self.embedding.weight.t()))

            min_encoding_indices = torch.argmin(d, dim=1) # idnex
            z_qi = self.embedding(min_encoding_indices).view(z_t.shape)
            loss = loss +  self.beta *torch.mean((z_qi.detach() - z_t)**2) +   torch.mean((z_qi - z_t.detach())**2) #前一项更新motif 后一项更新codebook
            index_list.append(min_encoding_indices)
            z_q.append(z_t + (z_qi - z_t).detach())

        # z_q = z_q.permute(0, 3, 1, 2)

        return z_q, loss,index_list