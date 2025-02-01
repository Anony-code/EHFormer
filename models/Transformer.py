import math
import numpy as np
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from models.simplehgn import myGAT


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.commitment_cost = nn.Parameter(torch.tensor(0.25))

        # self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # Flatten input
        input_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.embedding_dim)

        # Calculate distances to embedding vectors
        distances = (flat_inputs.pow(2).sum(1, keepdim=True)
                     - 2 * flat_inputs @ self.embedding.weight.t()
                     + self.embedding.weight.pow(2).sum(1, keepdim=True).t())
        
        # soft assignment
        weights = F.softmax(-distances, dim=1)  # Compute soft assignments
        quantized = weights @ self.embedding.weight  # Weighted average of codewords

        # # hard assignment
        # # Get the closest embedding indices
        # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        # encodings.scatter_(1, encoding_indices, 1)

        # # # Quantize
        # quantized = encodings @ self.embedding.weight
        quantized = quantized.view(*input_shape)

        # Loss to enforce commitment
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # vq_diversity_loss = diversity_loss(self.embedding.weight)

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # return self.embedding.weight, loss
        return quantized, loss



class GTLayer(nn.Module):
    def __init__(self, embeddings_dimension, nheads=2, dropout=0.5, temperature=1.0, num_quantized_embeddings=128, args=None):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''

        super(GTLayer, self).__init__()

        self.args = args
        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension
        self.dropout = dropout

        self.head_dim = self.embeddings_dimension // self.nheads

        self.temper = temperature

        self.linear_k = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.linear_v = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.linear_q = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        
        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.embeddings_dimension, bias=False)
        self.dropout_att = nn.Dropout(self.dropout)
        self.dropout_mlp = nn.Dropout(self.dropout)
        self.dropout_msa = nn.Dropout(self.dropout)
        self.dropout_ffn = nn.Dropout(self.dropout)

        self.activation = nn.LeakyReLU(0.2)

        self.FFN1 = nn.Linear(embeddings_dimension, embeddings_dimension)
        self.FFN2 = nn.Linear(embeddings_dimension, embeddings_dimension)
        self.LN1 = nn.LayerNorm(embeddings_dimension)
        self.LN2 = nn.LayerNorm(embeddings_dimension)


        self.vq_k = VectorQuantizer(num_quantized_embeddings, self.head_dim*self.nheads, commitment_cost=0.25)

    def forward(self, h, mask=None, e=1e-12):


        q = self.linear_q(h)
        k = self.linear_k(h)
        v = self.linear_v(h)
        batch_size = k.size()[0]

        # k, vq_loss = self.vq_k(k)

        q_ = q.reshape(batch_size, -1, self.nheads,
                       self.head_dim).transpose(1, 2)
        k_ = k.reshape(batch_size, -1, self.nheads,
                       self.head_dim).transpose(1, 2)
        v_ = v.reshape(batch_size, -1, self.nheads,
                       self.head_dim).transpose(1, 2)

        if self.args.vq_flag:
            h_new, vq_loss = self.vq_k(h)
            k_new = self.linear_k(h_new)
            v_new = self.linear_v(h_new)
            k_new_ = k_new.reshape(batch_size, -1, self.nheads,
            self.head_dim).transpose(1, 2)
            v_new_ = v_new.reshape(batch_size, -1, self.nheads,
            self.head_dim).transpose(1, 2)
            k_ = torch.cat([k_, k_new_], dim=2)
            v_ = torch.cat([v_, v_new_], dim=2)
        #############
        else:
            vq_loss = 0

        k_t = k_.transpose(2, 3)
        score = (q_ @ k_t) / math.sqrt(self.head_dim)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        score = F.softmax(score / self.temper, dim=-1)
        score = self.dropout_att(score)
        context = score @ v_
        ################

        h_sa = context.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.nheads)
        h_sa = self.activation(self.linear_final(h_sa))

        h_sa = self.dropout_msa(h_sa)
        h1 = self.LN1(h_sa + h)

        hf = self.activation(self.FFN1(h1))
        hf = self.dropout_mlp(hf)
        hf = self.FFN2(hf)
        hf = self.dropout_ffn(hf)

        h2 = self.LN2(h1+hf)
        return h2, vq_loss


class GT(nn.Module):
    def __init__(self, num_class, input_dimensions, vq_dim, meta_p, embeddings_dimension=64,  num_layers=8, nheads=2, dropout=0, temper=1.0, meta_vec=None, hg=None, coarsen_mat=None, args=None, hgnn_params=None, g=None, id_dim=64,  node2type=None):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''
        super(GT, self).__init__()

        self.args = args
        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        self.id_dim = id_dim
        self.dim_type = 5
        self.meta_p = meta_p
        self.node2type = node2type.to(args.device)
        self.hgnn = myGAT(g,
                          hgnn_params['dim'],
                          hgnn_params['num_etypes'],
                          hgnn_params['in_dim'],
                          hgnn_params['dim'],
                          embeddings_dimension,
                          hgnn_params['num_layers'],
                          hgnn_params['num_heads'],
                          F.elu,
                          hgnn_params['dropout'],
                          hgnn_params['dropout'],
                          0.05,
                          True,
                          0.05)
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, embeddings_dimension, bias=True) for in_dim in input_dimensions])
        self.dropout = dropout
        self.GTLayers = torch.nn.ModuleList()
        if self.args.meta_path_flag:
            for layer in range(self.num_layers):
                self.GTLayers.append(
                    GTLayer(self.embeddings_dimension*2+self.id_dim*2+self.meta_p*2, self.nheads, self.dropout, temperature=temper, num_quantized_embeddings=vq_dim, args=self.args))
            self.predictor = nn.Linear(embeddings_dimension*2+id_dim*2+self.meta_p*2, num_class, bias=False)
        else:
            for layer in range(self.num_layers):
                self.GTLayers.append(
                    GTLayer(self.embeddings_dimension*2+self.id_dim*2, self.nheads, self.dropout, temperature=temper, num_quantized_embeddings=vq_dim, args=self.args))
            self.predictor = nn.Linear(embeddings_dimension*2+id_dim*2, num_class, bias=False)

        self.hg = hg.to(args.device)
        self.coarsen_mat = coarsen_mat.to(args.device)
        self.global_token_flag = args.global_token_flag
        self.path_token_flag = args.path_token_flag
        self.node_token_flag = args.node_token_flag
        self.dataset = args.dataset
        self.target_node_id = nn.Embedding(1, id_dim)
        self.node_token_id = nn.Embedding(1, id_dim)
        # self.path_token_id = nn.Embedding(1, id_dim)
        # self.global_token_id = nn.Embedding(1, id_dim)
        self.type_id = nn.Embedding(11, self.dim_type)
        # self.type_id.weight.requires_grad = False
        # self.meta_vec = meta_vec.to(args.device)

        ############### meta-path
        self.meta_vec = nn.Embedding(num_embeddings=meta_vec.shape[0]+1, embedding_dim=meta_vec.shape[1], padding_idx=0)
        meta_vec = np.vstack([np.zeros((1, meta_vec.shape[1])), meta_vec])
        self.meta_vec.weight.data.copy_(torch.tensor(meta_vec).to(args.device))
        if args.dataset in ("ACM" or "DBLP"):
            self.meta_vec.weight.requires_grad = True
        else:
            self.meta_vec.weight.requires_grad = False

    def forward(self, features_list, e_feat, seqs, norm=False): ## add two sequences
        features_list = self.hgnn(features_list, e_feat)

        h = []
        h.append(features_list)

        h.append(torch.zeros(1, self.embeddings_dimension, device=h[0].device))
        h = torch.cat(h, 0)
        if self.args.meta_path_flag:
            graph_seq = torch.zeros([seqs.size(0), 2, 0, self.embeddings_dimension + self.id_dim+self.meta_p], device=h.device)
        else:
            graph_seq = torch.zeros([seqs.size(0), 2, 0, self.embeddings_dimension + self.id_dim], device=h.device)
            # graph_seq = torch.zeros([seqs.size(0), 0, self.embeddings_dimension + self.id_dim], device=h.device)
        

        # node token
        if self.node_token_flag:
            seqs = seqs[:,1:,:]
            seqs = seqs.to(torch.long)
            h_node = h[seqs]

            new_seqs = seqs+1
            ## meta-path
            if self.args.meta_path_flag:
                m_node = self.meta_vec(new_seqs.to(h[0].device))
                h_node = torch.cat([h_node, m_node], dim=-1)

            
            ########################
            node_token_id = self.node_token_id.weight.repeat([h_node.size(0), h_node.size(1), h_node.size(2), 1])
            h_node = torch.cat([h_node, node_token_id], dim=3)
            graph_seq = torch.cat([graph_seq, h_node], dim=2)
            # node_token_id = self.node_token_id.weight.repeat([h_node.size(0), h_node.size(1), 1])
            # h_node = torch.cat([h_node, node_token_id], dim=2)
            # graph_seq = torch.cat([graph_seq, h_node], dim=1)
            ########################
            
            #######################
            ##### debug change
            graph_seq = graph_seq.permute(0, 2, 1, 3)
            graph_seq = graph_seq.reshape(h_node.size(0), h_node.size(2), h_node.size(3)*2)
            #######################

        h = graph_seq
        vq = []
        for layer in range(self.num_layers):
            h, vq_loss = self.GTLayers[layer](h)
            vq.append(vq_loss)
        output = self.predictor(h[:, 0, :])

        if norm:
            output = output / (torch.norm(output, dim=1, keepdim=True)+1e-12)

        total_vq = sum(vq)

        return output, total_vq

