import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class AggreOutput(nn.Module):
    def __init__(self,kernel_size=-1,in_features=120,hidden_size=256,out_features=60):
        super(AggreOutput,self).__init__()
        self.kernel_size = kernel_size
        if kernel_size < 0:
            self.input_layer = nn.Linear(in_features=in_features,out_features=hidden_size)
            self.out_layer = nn.Linear(in_features=hidden_size,out_features=out_features)
        else:
            padding = kernel_size - 1
            self.padding_operator = nn.ConstantPad1d((padding, 0), 0)
            self.input_layer = nn.Conv1d(in_features, hidden_size, kernel_size=kernel_size, padding=0, bias=True)
            self.out_layer = nn.Conv1d(hidden_size,out_features,kernel_size=kernel_size,padding=int((kernel_size-1)/2),bias=False)

    def forward(self,ts_predict,event_predict):

        if self.kernel_size < 0:
            ts_predict = ts_predict.squeeze()
            event_predict = event_predict.squeeze()
            if len(ts_predict.shape) < 2:
                ts_predict = ts_predict.unsqueeze(0).contiguous()
            if len(event_predict.shape) < 2:
                event_predict = event_predict.unsqueeze(0).contiguous()
            # torch.cat((torch.randn(2,3),torch.randn(2,3)),dim=1).shape
            # axis=1
            input_tensor = torch.cat((ts_predict,event_predict),dim=1)
            input_tensor = input_tensor.contiguous()
            x1 = self.input_layer(input_tensor)
            x2 = self.out_layer(x1)
        else:
            # ts_input = self.padding_operator(ts_predict)
            # event_input = self.padding_operator(event_predict)
            # input_tensor = torch.concat([ts_predict,event_predict],axis=1)
            input_tensor = torch.cat((ts_predict, event_predict), dim=1)
            input_tensor = self.padding_operator(input_tensor)
            input_tensor = input_tensor.contiguous()
            x1 = self.input_layer(input_tensor)
            # x1_inter = self.padding_operator(x1)
            x2 = self.out_layer(x1)

        return x2




class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,qvk_kernel_size=5,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False,kernel_size=qvk_kernel_size),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix,kernel_size=qvk_kernel_size),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False,kernel_size=qvk_kernel_size),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,type=1):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,qvk_kernel_size=5,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False,kernel_size=qvk_kernel_size),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix,kernel_size=qvk_kernel_size),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False,kernel_size=qvk_kernel_size),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,type=1):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

class KAEInformer(nn. Module):
    def __init__(self, ts_enc_in, ts_dec_in, ts_c_out, ts_seq_len, ts_label_len, ts_out_len,
                  event_seq_len, event_label_len, event_out_len,
                 ts_factor=5, ts_d_model=512, ts_n_heads=8, ts_e_layers=3, ts_d_layers=2, ts_d_ff=512,
                 ts_dropout=0.0, ts_attn='prob', ts_embed='fixed',  ts_activation='gelu',
                 ts_distil=True, ts_mix=True, ts_qvk_kernel_size=5,event_factor=3,event_d_model=384, event_n_heads=8, event_e_layers=2, event_d_layers=1, event_d_ff=384,
                 event_dropout=0.0, event_attn='full', event_embed='fixed',  event_activation='gelu',
                 event_distil=True, event_mix=True,event_qvk_kernel_size=3,freq='h',output_attention=False,
                 device=torch.device('cuda:0')):
        super(KAEInformer, self).__init__()

        self.ts_informer = Informer(enc_in=ts_enc_in,dec_in=ts_dec_in,c_out=ts_c_out,seq_len=ts_seq_len,label_len=ts_label_len,
                                    out_len=ts_out_len,factor=ts_factor,d_model=ts_d_model,n_heads=ts_n_heads,
                                    e_layers=ts_e_layers,d_layers=ts_d_layers,d_ff=ts_d_ff,dropout=ts_dropout,
                                    attn=ts_attn,embed=ts_embed,freq=freq,activation=ts_activation,output_attention=output_attention,
                                    distil=ts_distil,mix=ts_mix,qvk_kernel_size=ts_qvk_kernel_size,device=device)

        self.event_informer = Informer(enc_in=ts_out_len,dec_in=ts_out_len,c_out=ts_out_len,seq_len=event_seq_len,
                                       label_len=event_label_len,out_len=event_out_len,factor=event_factor,d_model=event_d_model,
                                       n_heads=event_n_heads,e_layers=event_e_layers,d_layers=event_d_layers,d_ff=event_d_ff,
                                       dropout=event_dropout,attn=event_attn,embed=ts_embed,freq=freq,
                                       activation=event_activation,output_attention=output_attention,
                                       distil=event_distil,mix=event_mix,qvk_kernel_size=event_qvk_kernel_size,device=device)

    def forward(self,x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,type=1):
        if type <= 1:
            return self.ts_informer.forward(x_enc=x_enc,x_mark_enc=x_mark_enc,x_dec=x_dec,
                                            x_mark_dec=x_mark_dec,enc_self_mask=enc_self_mask,dec_self_mask=dec_self_mask,
                                            dec_enc_mask=dec_enc_mask)
        else:
            return self.event_informer.forward(x_enc=x_enc,x_mark_enc=x_mark_enc,x_dec=x_dec,
                                            x_mark_dec=x_mark_dec,enc_self_mask=enc_self_mask,dec_self_mask=dec_self_mask,
                                            dec_enc_mask=dec_enc_mask)


class CSEAInformer(nn. Module):
    def __init__(self, ts_enc_in, ts_dec_in, ts_c_out, ts_seq_len, ts_label_len, ts_out_len,
                  event_seq_len, event_label_len, event_out_len,
                 ts_factor=5, ts_d_model=512, ts_n_heads=8, ts_e_layers=3, ts_d_layers=2, ts_d_ff=512,
                 ts_dropout=0.0, ts_attn='prob', ts_embed='fixed',  ts_activation='gelu',
                 ts_distil=True, ts_mix=True, ts_qvk_kernel_size=5,event_factor=3,event_d_model=384, event_n_heads=8, event_e_layers=2, event_d_layers=1, event_d_ff=384,
                 event_dropout=0.0, event_attn='full', event_embed='fixed',  event_activation='gelu',
                 event_distil=True, event_mix=True,event_qvk_kernel_size=3,freq='h',output_attention=False,out_kernel_size=5,out_hidden_size=256,
                 device=torch.device('cuda:0')):
        super(CSEAInformer, self).__init__()

        if out_kernel_size < 0:
            self.final_output = AggreOutput(kernel_size=out_kernel_size,hidden_size=out_hidden_size,in_features=ts_out_len*2,out_features=ts_out_len)
        else:
            self.final_output = AggreOutput(kernel_size=out_kernel_size, hidden_size=out_hidden_size,in_features=event_out_len*2,out_features=event_out_len)

        self.ts_informer = Informer(enc_in=ts_enc_in,dec_in=ts_dec_in,c_out=ts_c_out,seq_len=ts_seq_len,label_len=ts_label_len,
                                    out_len=ts_out_len,factor=ts_factor,d_model=ts_d_model,n_heads=ts_n_heads,
                                    e_layers=ts_e_layers,d_layers=ts_d_layers,d_ff=ts_d_ff,dropout=ts_dropout,
                                    attn=ts_attn,embed=ts_embed,freq=freq,activation=ts_activation,output_attention=output_attention,
                                    distil=ts_distil,mix=ts_mix,qvk_kernel_size=ts_qvk_kernel_size,device=device)

        self.event_informer = Informer(enc_in=ts_out_len,dec_in=ts_out_len,c_out=ts_out_len,seq_len=event_seq_len,
                                       label_len=event_label_len,out_len=event_out_len,factor=event_factor,d_model=event_d_model,
                                       n_heads=event_n_heads,e_layers=event_e_layers,d_layers=event_d_layers,d_ff=event_d_ff,
                                       dropout=event_dropout,attn=event_attn,embed=ts_embed,freq=freq,
                                       activation=event_activation,output_attention=output_attention,
                                       distil=event_distil,mix=event_mix,qvk_kernel_size=event_qvk_kernel_size,device=device)

    def forward(self,x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,type=1):
        if type <= 1:
            return self.ts_informer.forward(x_enc=x_enc,x_mark_enc=x_mark_enc,x_dec=x_dec,
                                            x_mark_dec=x_mark_dec,enc_self_mask=enc_self_mask,dec_self_mask=dec_self_mask,
                                            dec_enc_mask=dec_enc_mask)
        elif type <= 2:
            return self.event_informer.forward(x_enc=x_enc,x_mark_enc=x_mark_enc,x_dec=x_dec,
                                            x_mark_dec=x_mark_dec,enc_self_mask=enc_self_mask,dec_self_mask=dec_self_mask,
                                            dec_enc_mask=dec_enc_mask)
        else:
            return self.final_output.forward(ts_predict=x_enc,event_predict=x_dec)

