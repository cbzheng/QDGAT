import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from tools import allennlp as util
import dgl

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)

# https://github.com/dmlc/dgl/blob/master/examples/pytorch/hgt/model.py 
class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)

class HGT(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1, hgt_n_heads=2, use_norm=True):
        super(HGT, self).__init__()
        print('use hgt')
        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self.node_dict = {
            'd': 0,
            'q': 1,
        }

        self.edge_dict = {
            'd_lg_q': 0,
            'd_lg_d': 1,
            'q_lg_q': 2,
            'q_lg_d': 3,
            'd_sm_q': 4,
            'd_sm_d': 5,
            'q_sm_q': 6,
            'q_sm_d': 7,
        }

        # HGT
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(self.node_dict)):
            self.adapt_ws.append(nn.Linear(node_dim, node_dim))
        self.hgt =HGTLayer(node_dim, node_dim, self.node_dict, self.edge_dict, hgt_n_heads, use_norm = use_norm)
        self.out = nn.Linear(node_dim, node_dim)


    def forward(self, d_node, q_node, d_node_mask, q_node_mask, graph, extra_factor=None):

        d_node_len = d_node.size(1)
        q_node_len = q_node.size(1)

        diagmat = torch.diagflat(torch.ones(d_node.size(1), dtype=torch.long, device=d_node.device))
        diagmat = diagmat.unsqueeze(0).expand(d_node.size(0), -1, -1)
        dd_graph = d_node_mask.unsqueeze(1) * d_node_mask.unsqueeze(-1) * (1 - diagmat)
        # larger than
        dd_graph_left = dd_graph * graph[:, :d_node_len, :d_node_len]
        # smaller than
        dd_graph_right = dd_graph * (1 - graph[:, :d_node_len, :d_node_len])

        diagmat = torch.diagflat(torch.ones(q_node.size(1), dtype=torch.long, device=q_node.device))
        diagmat = diagmat.unsqueeze(0).expand(q_node.size(0), -1, -1)
        qq_graph = q_node_mask.unsqueeze(1) * q_node_mask.unsqueeze(-1) * (1 - diagmat)
        qq_graph_left = qq_graph * graph[:, d_node_len:, d_node_len:]
        qq_graph_right = qq_graph * (1 - graph[:, d_node_len:, d_node_len:])

        dq_graph = d_node_mask.unsqueeze(-1) * q_node_mask.unsqueeze(1)
        dq_graph_left = dq_graph * graph[:, :d_node_len, d_node_len:]
        dq_graph_right = dq_graph * (1 - graph[:, :d_node_len, d_node_len:])

        qd_graph = q_node_mask.unsqueeze(-1) * d_node_mask.unsqueeze(1)
        qd_graph_left = qd_graph * graph[:, d_node_len:, :d_node_len]
        qd_graph_right = qd_graph * (1 - graph[:, d_node_len:, :d_node_len])

        # construct heterogeneous graph using dgl
        q_batch_embs = []
        d_batch_embs = []

        for batch_idx in range(d_node.size(0)):
            d_lg_d = torch.nonzero(dd_graph_left[batch_idx]).transpose(0, 1)
            q_lg_q = torch.nonzero(qq_graph_left[batch_idx]).transpose(0, 1)
            d_lg_q = torch.nonzero(dq_graph_left[batch_idx]).transpose(0, 1)
            q_lg_d = torch.nonzero(qd_graph_left[batch_idx]).transpose(0, 1)
            d_sm_d = torch.nonzero(dd_graph_right[batch_idx]).transpose(0, 1)
            q_sm_q = torch.nonzero(qq_graph_right[batch_idx]).transpose(0, 1)
            d_sm_q = torch.nonzero(dq_graph_right[batch_idx]).transpose(0, 1)
            q_sm_d = torch.nonzero(qd_graph_right[batch_idx]).transpose(0, 1)
            graph_data = {
                ('d', 'd_lg_q','q'): (d_lg_q[0], d_lg_q[1]),
                ('q', 'q_lg_d', 'd'): (q_lg_d[0], q_lg_d[1]),
                ('d', 'd_lg_d', 'd'): (d_lg_d[0], d_lg_d[1]),
                ('q', 'q_lg_q', 'q'): (q_lg_q[0], q_lg_q[1]),
                ('d', 'd_sm_q','q'): (d_sm_q[0], d_sm_q[1]),
                ('q', 'q_sm_d', 'd'): (q_sm_d[0], q_sm_d[1]),
                ('d', 'd_sm_d', 'd'): (d_sm_d[0], d_sm_d[1]),
                ('q', 'q_sm_q', 'q'): (q_sm_q[0], q_sm_q[1]),
            }
            g = dgl.heterograph(graph_data)
            d_node_indices = torch.nonzero(d_node_mask[batch_idx]).squeeze(1)
            q_node_indices = torch.nonzero(q_node_mask[batch_idx]).squeeze(1)
            g.nodes['d'].data['inp'] = d_node[batch_idx][d_node_indices]
            g.nodes['q'].data['inp'] = q_node[batch_idx][q_node_indices]

            h = {}
            for ntype in g.ntypes:
                n_id = self.node_dict[ntype]
                h[ntype] = F.gelu(self.adapt_ws[n_id](g.nodes[ntype].data['inp']))
            h = self.hgt(g, h)
            d_out = self.out(h['d'])
            q_out = self.out(h['q'])

            pad_size=d_node_len - d_out.shape[0]
            paddings = torch.zeros(pad_size, self.node_dim)
            d_out = torch.cat((d_out, paddings.to(d_out.device)))

            pad_size = q_node_len - q_out.shape[0]
            paddings = torch.zeros(pad_size, self.node_dim)
            q_out = torch.cat((q_out, paddings.to(q_out.device)))

            q_batch_embs.append(q_out)
            d_batch_embs.append(d_out)

        q_node = torch.stack(q_batch_embs, dim=0)
        d_node = torch.stack(d_batch_embs, dim=0)

        return d_node, q_node, 'useless', 'useless'


class GCN(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(GCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._dd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)

        self._dd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)

    def forward(self, d_node, q_node, d_node_mask, q_node_mask, graph, extra_factor=None):

        d_node_len = d_node.size(1)
        q_node_len = q_node.size(1)

        diagmat = torch.diagflat(torch.ones(d_node.size(1), dtype=torch.long, device=d_node.device))
        diagmat = diagmat.unsqueeze(0).expand(d_node.size(0), -1, -1)
        dd_graph = d_node_mask.unsqueeze(1) * d_node_mask.unsqueeze(-1) * (1 - diagmat)
        # larger than
        dd_graph_left = dd_graph * graph[:, :d_node_len, :d_node_len]
        # smaller than
        dd_graph_right = dd_graph * (1 - graph[:, :d_node_len, :d_node_len])

        diagmat = torch.diagflat(torch.ones(q_node.size(1), dtype=torch.long, device=q_node.device))
        diagmat = diagmat.unsqueeze(0).expand(q_node.size(0), -1, -1)
        qq_graph = q_node_mask.unsqueeze(1) * q_node_mask.unsqueeze(-1) * (1 - diagmat)
        qq_graph_left = qq_graph * graph[:, d_node_len:, d_node_len:]
        qq_graph_right = qq_graph * (1 - graph[:, d_node_len:, d_node_len:])

        dq_graph = d_node_mask.unsqueeze(-1) * q_node_mask.unsqueeze(1)
        dq_graph_left = dq_graph * graph[:, :d_node_len, d_node_len:]
        dq_graph_right = dq_graph * (1 - graph[:, :d_node_len, d_node_len:])

        qd_graph = q_node_mask.unsqueeze(-1) * d_node_mask.unsqueeze(1)
        qd_graph_left = qd_graph * graph[:, d_node_len:, :d_node_len]
        qd_graph_right = qd_graph * (1 - graph[:, d_node_len:, :d_node_len])

        d_node_neighbor_num = dd_graph_left.sum(-1) + dd_graph_right.sum(-1) + dq_graph_left.sum(-1) + dq_graph_right.sum(-1)
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = util.replace_masked_values(d_node_neighbor_num.float(), d_node_neighbor_num_mask, 1)

        q_node_neighbor_num = qq_graph_left.sum(-1) + qq_graph_right.sum(-1) + qd_graph_left.sum(-1) + qd_graph_right.sum(-1)
        q_node_neighbor_num_mask = (q_node_neighbor_num >= 1).long()
        q_node_neighbor_num = util.replace_masked_values(q_node_neighbor_num.float(), q_node_neighbor_num_mask, 1)


        all_d_weight, all_q_weight = [], []
        for step in range(self.iteration_steps):
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(d_node)).squeeze(-1)
                q_node_weight = torch.sigmoid(self._node_weight_fc(q_node)).squeeze(-1)
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((d_node, extra_factor), dim=-1))).squeeze(-1)
                q_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((q_node, extra_factor), dim=-1))).squeeze(-1)

            all_d_weight.append(d_node_weight)
            all_q_weight.append(q_node_weight)

            self_d_node_info = self._self_node_fc(d_node)
            self_q_node_info = self._self_node_fc(q_node)

            dd_node_info_left = self._dd_node_fc_left(d_node)
            qd_node_info_left = self._qd_node_fc_left(d_node)
            qq_node_info_left = self._qq_node_fc_left(q_node)
            dq_node_info_left = self._dq_node_fc_left(q_node)

            dd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dd_graph_left,
                    0)

            qd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qd_graph_left,
                    0)

            qq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qq_graph_left,
                    0)

            dq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dq_graph_left,
                    0)

            dd_node_info_left = torch.matmul(dd_node_weight, dd_node_info_left)
            qd_node_info_left = torch.matmul(qd_node_weight, qd_node_info_left)
            qq_node_info_left = torch.matmul(qq_node_weight, qq_node_info_left)
            dq_node_info_left = torch.matmul(dq_node_weight, dq_node_info_left)


            dd_node_info_right = self._dd_node_fc_right(d_node)
            qd_node_info_right = self._qd_node_fc_right(d_node)
            qq_node_info_right = self._qq_node_fc_right(q_node)
            dq_node_info_right = self._dq_node_fc_right(q_node)

            dd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dd_graph_right,
                    0)

            qd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qd_graph_right,
                    0)

            qq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qq_graph_right,
                    0)

            dq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dq_graph_right,
                    0)

            dd_node_info_right = torch.matmul(dd_node_weight, dd_node_info_right)
            qd_node_info_right = torch.matmul(qd_node_weight, qd_node_info_right)
            qq_node_info_right = torch.matmul(qq_node_weight, qq_node_info_right)
            dq_node_info_right = torch.matmul(dq_node_weight, dq_node_info_right)


            agg_d_node_info = (dd_node_info_left + dd_node_info_right + dq_node_info_left + dq_node_info_right) / d_node_neighbor_num.unsqueeze(-1)
            agg_q_node_info = (qq_node_info_left + qq_node_info_right + qd_node_info_left + qd_node_info_right) / q_node_neighbor_num.unsqueeze(-1)

            d_node = F.relu(self_d_node_info + agg_d_node_info)
            q_node = F.relu(self_q_node_info + agg_q_node_info)


        all_d_weight = [weight.unsqueeze(1) for weight in all_d_weight]
        all_q_weight = [weight.unsqueeze(1) for weight in all_q_weight]

        all_d_weight = torch.cat(all_d_weight, dim=1)
        all_q_weight = torch.cat(all_q_weight, dim=1)

        return d_node, q_node, d_node_weight, q_node_weight

        #torch.Size([4, 1, 43])