from rgcn.layers import RelationalGraphConvolutionLP, DistMult
from rgcn.utils import add_inverse_and_self, select_w_init
import torch.nn.functional as F
from torch import nn
import torch

torch.set_printoptions(precision=5)

######################################################################################
# Models for Experiment Reproduction
######################################################################################


class LinkPredictor(nn.Module):
    """ Link Prediction using an RGCN-based encoder and DistMult decoder """
    def __init__(self,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 encoder_config=None,
                 decoder_config=None):
        super(LinkPredictor, self).__init__()

        # Encoder config
        nemb = encoder_config["node_embedding"] if "node_embedding" in encoder_config else None
        nhid1 = encoder_config["hidden1_size"] if "hidden1_size" in encoder_config else None
        nhid2 = encoder_config["hidden2_size"] if "hidden2_size" in encoder_config else None
        rgcn_layers = encoder_config["num_layers"] if "num_layers" in encoder_config else 2
        edge_dropout = encoder_config["edge_dropout"] if "edge_dropout" in encoder_config else None
        decomposition = encoder_config["decomposition"] if "decomposition" in encoder_config else None
        encoder_w_init = encoder_config["weight_init"] if "weight_init" in encoder_config else None
        encoder_gain = encoder_config["include_gain"] if "include_gain" in encoder_config else False
        encoder_b_init = encoder_config["bias_init"] if "bias_init" in encoder_config else None

        # Decoder config
        decoder_l2_type = decoder_config["l2_penalty_type"] if "l2_penalty_type" in decoder_config else None
        decoder_l2 = decoder_config["l2_penalty"] if "l2_penalty" in decoder_config else None
        decoder_w_init = decoder_config["weight_init"] if "weight_init" in decoder_config else None
        decoder_gain = decoder_config["include_gain"] if "include_gain" in decoder_config else False
        decoder_b_init = decoder_config["bias_init"] if "bias_init" in decoder_config else None

        assert (nnodes is not None or nrel is not None or nhid1 is not None), \
            "The following must be specified: number of nodes, number of relations and output dimension!"
        assert 0 < rgcn_layers < 3, "Only supports the following number of convolution layers: 1 and 2."

        self.num_nodes = nnodes
        self.num_rels = nrel
        self.rgcn_layers = rgcn_layers
        self.nemb = nemb
        self.decoder_l2_type = decoder_l2_type
        self.decoder_l2 = decoder_l2

        self.node_embeddings = nn.Parameter(torch.FloatTensor(nnodes, nemb))
        self.node_embeddings_bias = nn.Parameter(torch.zeros(1, nemb))
        init = select_w_init(encoder_w_init)
        init(self.node_embeddings)
        # Checkpoint 1
        # print('self.node_embeddings')
        # print('min', torch.min(self.node_embeddings))
        # print('max', torch.max(self.node_embeddings))
        # print('mean', torch.mean(self.node_embeddings))
        # print('std', torch.std(self.node_embeddings))
        # print('size', self.node_embeddings.size())

        # Encoder
        self.rgc1 = RelationalGraphConvolutionLP(
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nemb,
            out_features=nhid1,
            edge_dropout=edge_dropout,
            decomposition=decomposition,
            vertical_stacking=False,
            w_init=encoder_w_init,
            w_gain=encoder_gain,
            b_init=encoder_b_init
        )
        if rgcn_layers == 2:
            self.rgc2 = RelationalGraphConvolutionLP(
                num_nodes=nnodes,
                num_relations=nrel * 2 + 1,
                in_features=nhid1,
                out_features=nhid2,
                edge_dropout=edge_dropout,
                decomposition=decomposition,
                vertical_stacking=False,
                w_init=encoder_w_init,
                w_gain=encoder_gain,
                b_init=encoder_b_init
            )

        # Decoder
        self.scoring_function = DistMult(nrel, nemb, nnodes, nrel, decoder_w_init, decoder_gain, decoder_b_init)
        #self.scoring_function = TransC(nrel, nemb, nnodes, nrel, decoder_w_init, decoder_gain, decoder_b_init)

    def compute_penalty(self, batch, x):
        """ Compute L2 penalty for decoder """
        if self.decoder_l2 == 0.0:
            return 0

        if self.decoder_l2_type == 'schlichtkrull-l2':
            return self.scoring_function.s_penalty(batch, x)
        else:
            return self.scoring_function.relations.pow(2).sum()

    def forward(self, graph, triples):
        """ Embed relational graph and then compute score """

        if self.nemb is not None:
            x = self.node_embeddings + self.node_embeddings_bias
            x = torch.nn.functional.relu(x)
            # Checkpoint 1
            # print('min', torch.min(x))
            # print('max', torch.max(x))
            # print('mean', torch.mean(x))
            # print('std', torch.std(x))
            # print('size', x.size())
            x = self.rgc1(graph, features=x)
        else:
            x = self.rgc1(graph)

        if self.rgcn_layers == 2:
            x = F.relu(x)
            x = self.rgc2(graph, features=x)

        scores = self.scoring_function(triples, x)
        # Checkpoint 7
        print('min', torch.min(scores))
        print('max', torch.max(scores))
        print('mean', torch.mean(scores))
        print('std', torch.std(scores))
        print('size', scores.size())
        exit()
        penalty = self.compute_penalty(triples, x)
        return scores, penalty