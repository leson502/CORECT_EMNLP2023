import torch
import torch.nn as nn
import torch.nn.functional as F

from .Classifier import Classifier
from .UnimodalEncoder import UnimodalEncoder
from .CrossmodalNet import CrossmodalNet
from .GraphModel import GraphModel
from .functions import multi_concat, feature_packing
import corect

log = corect.utils.get_logger()

class CORECT(nn.Module):
    def __init__(self, args):
        super(CORECT, self).__init__()

        self.args = args
        self.wp = args.wp
        self.wf = args.wf
        self.modalities = args.modalities
        self.n_modals = len(self.modalities)
        self.use_speaker = args.use_speaker
        g_dim = args.hidden_size
        h_dim = args.hidden_size

        ic_dim = 0
        if not args.no_gnn:
            ic_dim = h_dim * self.n_modals

            if not args.use_graph_transformer and (args.gcn_conv == "gat_gcn" or args.gcn_conv == "gcn_gat"):
                ic_dim = ic_dim * 2

            if args.use_graph_transformer:
                ic_dim *= args.graph_transformer_nheads
        
        if args.use_crossmodal and self.n_modals > 1:
            ic_dim += h_dim * self.n_modals * (self.n_modals - 1)

        if self.args.no_gnn and (not self.args.use_crossmodal or self.n_modals == 1):
            ic_dim = h_dim * self.n_modals

        
        a_dim = args.dataset_embedding_dims[args.dataset]['a']
        t_dim = args.dataset_embedding_dims[args.dataset]['t']
        v_dim = args.dataset_embedding_dims[args.dataset]['v']
        
        dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
        }

        dataset_speaker_dict = {
            "iemocap": 2,
            "iemocap_4": 2,
            "mosei":1,
        }
        
        
        tag_size = len(dataset_label_dict[args.dataset])
        self.n_speakers = dataset_speaker_dict[args.dataset]

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device


        self.encoder = UnimodalEncoder(a_dim, t_dim, v_dim, g_dim, args)
        self.speaker_embedding = nn.Embedding(self.n_speakers, g_dim)

        print(f"{args.dataset} speakers: {self.n_speakers}")
        if not args.no_gnn:
            self.graph_model = GraphModel(g_dim, h_dim, h_dim, self.device, args)
            print('CORECT --> Use GNN')

        if args.use_crossmodal and self.n_modals > 1:
            self.crossmodal = CrossmodalNet(g_dim, args)
            print('CORECT --> Use Crossmodal')
        elif self.n_modals == 1:
            print('CORECT --> Crossmodal not available when number of modalitiy is 1')

        self.clf = Classifier(ic_dim, h_dim, tag_size, args)

        self.rlog = {}


    def represent(self, data):

        # Encoding multimodal feature
        a = data['audio_tensor'] if 'a' in self.modalities else None
        t = data['text_tensor'] if 't' in self.modalities else None
        v = data['visual_tensor'] if 'v' in self.modalities else None

        a, t, v = self.encoder(a, t, v, data['text_len_tensor'])


        # Speaker embedding
        if self.use_speaker:
            emb = self.speaker_embedding(data['speaker_tensor'])
            a = a + emb if a != None else None
            t = t + emb if t != None else None
            v = v + emb if v != None else None

        # Graph construct
        multimodal_features = []

        if a != None:
            multimodal_features.append(a)
        if t != None:
            multimodal_features.append(t)
        if v != None:
            multimodal_features.append(v)

        out_encode = feature_packing(multimodal_features, data['text_len_tensor'])
        out_encode = multi_concat(out_encode, data['text_len_tensor'], self.n_modals)

        out = []

        if not self.args.no_gnn:
            out_graph = self.graph_model(multimodal_features, data['text_len_tensor'])
            out.append(out_graph)


        if self.args.use_crossmodal and self.n_modals > 1:
            out_cr = self.crossmodal(multimodal_features)

            out_cr = out_cr.permute(1, 0, 2)
            lengths = data['text_len_tensor']
            batch_size = lengths.size(0)
            cr_feat = []
            for j in range(batch_size):
                cur_len = lengths[j].item()
                cr_feat.append(out_cr[j,:cur_len])

            cr_feat = torch.cat(cr_feat, dim=0).to(self.device)
            out.append(cr_feat)
        
        if self.args.no_gnn and (not self.args.use_crossmodal or self.n_modals == 1):
            out = out_encode
        else:
            out = torch.cat(out, dim=-1)

        return out

    def forward(self, data):
        graph_out = self.represent(data)
        out = self.clf(graph_out, data["text_len_tensor"])

        return out
    
    def get_loss(self, data):
        graph_out = self.represent(data)
        loss = self.clf.get_loss(
                graph_out, data["label_tensor"], data["text_len_tensor"])
        
        return loss

    def get_log(self):
        return self.rlog


        