import torch
from torch import nn
import torch.nn.functional as F
from model_part import Encoder, Decoder
from torch.nn.init import xavier_normal_


class TFNFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.post_fusion_dropout = nn.Dropout(p=args['dropout'])
        self.post_fusion_layer_1 = nn.Linear((args['h_dim'] + 1) * (args['h_dim'] + 1) * (args['h_dim'] + 1),
                                             args['h_dim'] // 2)
        self.post_fusion_layer_2 = nn.Linear(args['h_dim'] // 2, args['h_dim'] // 4)
        self.post_fusion_layer_3 = nn.Linear(args['h_dim'] // 4, 1)

    def forward(self, v, a, t):
        args = self.args
        batch_size = a.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(a).to(a.device)
        _audio_h = torch.cat((add_one, a), dim=1)
        _video_h = torch.cat((add_one, v), dim=1)
        _text_h = torch.cat((add_one, t), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))

        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (args['h_dim'] + 1) * (args['h_dim'] + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped), inplace=True)
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1), inplace=True)
        return self.post_fusion_layer_3(post_fusion_y_2)


class TFN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.v_enc = Encoder(args['v_dim'], args['h_dim'], args['dropout'])
        self.a_enc = Encoder(args['a_dim'], args['h_dim'], args['dropout'])
        self.t_enc = Encoder(args['t_dim'], args['h_dim'], args['dropout'])
        self.tfn = TFNFusion(args)

    def forward(self, fd):
        v, a = fd['v'], fd['a']
        t = fd['t'] if self.args['use_live_text'] else fd['p']
        a = a.mean(1)
        t = t.mean(1)
        v = self.v_enc(v)
        a = self.a_enc(a)
        t = self.t_enc(t)
        output = self.tfn(v, a, t)

        return output.squeeze(-1)


class LMF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.v_enc = Encoder(args['v_dim'], args['h_dim'], args['dropout'])
        self.a_enc = Encoder(args['a_dim'], args['h_dim'], args['dropout'])
        self.t_enc = Encoder(args['t_dim'], args['h_dim'], args['dropout'])

        self.rank = 64
        self.audio_factor = nn.Parameter(torch.Tensor(self.rank, args['h_dim'] + 1, 1))
        self.video_factor = nn.Parameter(torch.Tensor(self.rank, args['h_dim'] + 1, 1))
        self.text_factor = nn.Parameter(torch.Tensor(self.rank, args['h_dim'] + 1, 1))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, 1))
        # init teh factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, fd):
        args = self.args
        v, a = fd['v'], fd['a']
        t = fd['t'] if args['use_live_text'] else fd['p']
        a = a.mean(1)
        t = t.mean(1)
        v = self.v_enc(v)
        a = self.a_enc(a)
        t = self.t_enc(t)
        batch_size = a.data.shape[0]

        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(a).to(a.device)
        _audio_h = torch.cat((add_one, a), dim=1)
        _video_h = torch.cat((add_one, v), dim=1)
        _text_h = torch.cat((add_one, t), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, 1).squeeze(-1)

        return output

class MISA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cv_enc = Encoder(args['v_dim'], args['h_dim'], args['dropout'])
        self.ca_enc = Encoder(args['a_dim'], args['h_dim'], args['dropout'])
        self.ct_enc = Encoder(args['t_dim'], args['h_dim'], args['dropout'])
        self.pv_enc = Encoder(args['v_dim'], args['h_dim'], args['dropout'])
        self.pa_enc = Encoder(args['a_dim'], args['h_dim'], args['dropout'])
        self.pt_enc = Encoder(args['t_dim'], args['h_dim'], args['dropout'])
        # reconstruct
        self.v_dec = Decoder(2 * args['h_dim'], args['v_dim'], args['dropout'])
        self.a_dec = Decoder(2 * args['h_dim'], args['a_dim'], args['dropout'])
        self.t_dec = Decoder(2 * args['h_dim'], args['t_dim'], args['dropout'])

        self.fusion = nn.Linear(6 * args['h_dim'], 1)

    def forward(self, fd):
        v, a = fd['v'], fd['a']
        t = fd['t'] if self.args['use_live_text'] else fd['p']
        a = a.mean(1)
        t = t.mean(1)
        self.ov, self.oa, self.ot = v, a, t
        self.cv = self.cv_enc(v)
        self.ca = self.ca_enc(a)
        self.ct = self.ct_enc(t)
        self.pv = self.pv_enc(v)
        self.pa = self.pa_enc(a)
        self.pt = self.pt_enc(t)
        # reconstruct
        self.rv = self.v_dec(torch.cat([self.cv, self.pv], dim=-1))
        self.ra = self.a_dec(torch.cat([self.ca, self.pa], dim=-1))
        self.rt = self.t_dec(torch.cat([self.ct, self.pt], dim=-1))

        self.dif_pairs = [
            (self.pt, self.ct),
            (self.pv, self.cv),
            (self.pa, self.ca),
            (self.pa, self.pt),
            (self.pa, self.pv),
            (self.pt, self.pv),
        ]
        self.cmd_pairs = [
            (self.ct, self.cv),
            (self.ct, self.ca),
            (self.ca, self.cv),
        ]
        self.recon_pairs = [
            (self.ov, self.rv),
            (self.oa, self.ra),
            (self.ot, self.rt),
        ]

        return self.fusion(torch.cat([self.cv, self.ca, self.ct, self.pv, self.pa, self.pt], dim=-1)).squeeze(-1)


class LAD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.s_linguistic = Encoder(args['t_dim'], args['h_dim'], args['dropout'])
        self.a_linguistic = Encoder(args['a_dim'], args['h_dim'], args['dropout'])
        self.a_acoustic = Encoder(args['a_dim'], args['h_dim'], args['dropout'])
        self.t_linguistic = Encoder(args['t_dim'], args['h_dim'], args['dropout'])
        self.t_acoustic = Encoder(args['t_dim'], args['h_dim'], args['dropout'])
        self.v_enc = Encoder(args['v_dim'], args['h_dim'], args['dropout'])

        # reconstruct
        self.a_dec = Decoder(2 * args['h_dim'], args['a_dim'], args['dropout'])
        self.t_dec = Decoder(2 * args['h_dim'], args['t_dim'], args['dropout'])
        self.s_dec = Decoder(args['h_dim'], args['t_dim'], args['dropout'])
        self.v_dec = Decoder(args['h_dim'], args['v_dim'], args['dropout'])

        self.fusion = nn.Linear(3 * args['h_dim'], 1)

    def forward(self, fd):
        v, a, t, p = fd['v'], fd['a'], fd['t'], fd['p']
        a = a.mean(1)
        t = t.mean(1)
        p = p.mean(1)
        self.oa, self.ot, self.op, self.ov = a, t, p, v
        self.al = self.a_linguistic(a)
        self.aa = self.a_acoustic(a)
        self.tl = self.t_linguistic(t)
        self.ta = self.t_acoustic(t)
        self.vh = self.v_enc(v)
        self.ph = self.s_linguistic(p)
        # reconstruct
        self.ra = self.a_dec(torch.cat([self.al, self.aa], dim=-1))
        self.rt = self.t_dec(torch.cat([self.tl, self.ta], dim=-1))
        self.rp = self.s_dec(self.ph)
        self.rv = self.v_dec(self.vh)

        self.dif_pairs = [
            (self.al, self.aa),
            (self.tl, self.ta),
        ]
        self.cmd_pairs = [
            (self.ph, self.al),
            (self.ph, self.tl),
            (self.al, self.tl),
        ]
        self.recon_pairs = [
            (self.oa, self.ra),
            (self.ot, self.rt),
            (self.op, self.rp),
            (self.ov, self.rv),
        ]

        return self.fusion(torch.cat([self.vh, self.aa, self.ta], dim=-1))
