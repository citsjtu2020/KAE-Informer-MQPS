from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred,Dataset_QPS,getRandomIndex
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack,KAEInformer,CSEAInformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import math
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
            'kaeinformer': KAEInformer,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads,
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.args.qvk_kernel_size,
                self.device
            ).float()
        elif 'kae' in self.args.model:
            # parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
            # parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
            # parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
            #
            # parser.add_argument('--event_seq_len', type=int, default=96, help='input sequence length of Informer encoder')
            # parser.add_argument('--event_label_len', type=int, default=48, help='start token length of Informer decoder')
            # parser.add_argument('--event_pred_len', type=int, default=24, help='prediction sequence length')
            model = KAEInformer(ts_enc_in=self.args.enc_in,ts_dec_in=self.args.dec_in,
                                ts_c_out=self.args.c_out,ts_seq_len=self.args.seq_len,
                                ts_label_len=self.args.label_len,ts_out_len=self.args.pred_len,
                                event_seq_len=self.args.event_seq_len,event_label_len=self.args.event_label_len,
                                event_out_len=self.args.event_pred_len,
                                ts_factor=self.args.event_factor,ts_d_model=self.args.d_model,ts_n_heads=self.args.n_heads,
                                ts_e_layers=self.args.e_layers,ts_d_layers=self.args.d_layers,ts_d_ff=self.args.d_ff,
                                ts_dropout=self.args.dropout,ts_attn=self.args.attn,ts_embed=self.args.embed,ts_activation=self.args.activation,
                                ts_distil=self.args.distil,ts_mix=self.args.mix,ts_qvk_kernel_size=self.args.qvk_kernel_size,
                                event_factor=self.args.event_factor,event_d_model=self.args.event_d_model,event_n_heads=self.args.event_n_heads,
                                event_e_layers=self.args.event_e_layers,event_d_layers=self.args.event_d_layers,
                                event_d_ff=self.args.event_d_ff,event_dropout=self.args.dropout,event_attn=self.args.event_attn,event_embed=self.args.embed,
                                event_activation=self.args.event_activation,event_distil=self.args.event_distil,event_mix=self.args.mix,
                                event_qvk_kernel_size=self.args.event_qvk_kernel_size,freq=self.args.freq,output_attention=self.args.output_attention,
                                device=self.device)
        elif 'csea' in self.args.model:
            model = CSEAInformer(
                ts_enc_in=self.args.enc_in, ts_dec_in=self.args.dec_in,
                ts_c_out=self.args.c_out, ts_seq_len=self.args.seq_len,
                ts_label_len=self.args.label_len, ts_out_len=self.args.pred_len,
                event_seq_len=self.args.event_seq_len, event_label_len=self.args.event_label_len,
                event_out_len=self.args.event_pred_len,
                ts_factor=self.args.event_factor, ts_d_model=self.args.d_model, ts_n_heads=self.args.n_heads,
                ts_e_layers=self.args.e_layers, ts_d_layers=self.args.d_layers, ts_d_ff=self.args.d_ff,
                ts_dropout=self.args.dropout, ts_attn=self.args.attn, ts_embed=self.args.embed,
                ts_activation=self.args.activation,
                ts_distil=self.args.distil, ts_mix=self.args.mix, ts_qvk_kernel_size=self.args.qvk_kernel_size,
                event_factor=self.args.event_factor, event_d_model=self.args.event_d_model,
                event_n_heads=self.args.event_n_heads,
                event_e_layers=self.args.event_e_layers, event_d_layers=self.args.event_d_layers,
                event_d_ff=self.args.event_d_ff, event_dropout=self.args.dropout, event_attn=self.args.event_attn,
                event_embed=self.args.embed,
                event_activation=self.args.event_activation, event_distil=self.args.event_distil,
                event_mix=self.args.mix,
                event_qvk_kernel_size=self.args.event_qvk_kernel_size, freq=self.args.freq,
                output_attention=self.args.output_attention,out_hidden_size=self.args.out_hidden_size,out_kernel_size=self.args.out_kernel_size,
                device=self.device
            )
        else:
            model = Informer(
                enc_in=self.args.enc_in,dec_in=self.args.dec_in,c_out=self.args.c_out,seq_len=self.args.seq_len,
                label_len=self.args.label_len,out_len=self.args.pred_len,factor=self.args.factor,d_model=self.args.d_model,
                n_heads=self.args.n_heads,e_layers=self.args.e_layers,d_layers=self.args.d_layers,d_ff=self.args.d_ff,
                dropout=self.args.dropout,attn=self.args.attn,embed=self.args.embed,freq=self.args.freq,activation=self.args.activation,output_attention=self.args.output_attention,
                qvk_kernel_size=self.args.qvk_kernel_size,device=self.device
            )
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'qps': Dataset_QPS
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        if self.args.data != 'qps':
            data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
            )
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)

            return data_set, data_loader
        else:

            valid_index,train_index = getRandomIndex(self.args.data_item_size,x=math.floor(args.data_item_size*0.1))
            data_set_train = Dataset_QPS(root_path=self.args.root_path,flag='train',
                                    ts_size=[args.seq_len, args.label_len, args.pred_len],
                                    event_size=[args.event_seq_len,args.event_label_len,args.event_pred_len],
                                    app_group=args.app,freq=freq,timeenc=timeenc,using_index=train_index,sample_type=self.args.sample_type,type=self.args.data_type)

            data_loader_train = DataLoader(
                data_set_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True
            )
            data_set_valid = Dataset_QPS(root_path=self.args.root_path,flag='val',
                                    ts_size=[args.seq_len, args.label_len, args.pred_len],
                                    event_size=[args.event_seq_len,args.event_label_len,args.event_pred_len],
                                    app_group=args.app,freq=freq,timeenc=timeenc,using_index=valid_index,sample_type=self.args.sample_type,type=self.args.data_type)

            data_loader_valid = DataLoader(
                data_set_valid,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True)

            print(data_set_train.ts_scaler.std)
            print(data_set_train.event_scaler.std)

            data_set_test = Dataset_QPS(root_path=self.args.root_path, flag='test',
                                         ts_size=[args.seq_len, args.label_len, args.pred_len],
                                         event_size=[args.event_seq_len, args.event_label_len, args.event_pred_len],
                                         app_group=args.app, freq=freq, timeenc=timeenc,
                                        using_index=None,scale_type=2,
                                        scale_ts_std=data_set_train.ts_scaler.std,
                                        scale_ts_mean=data_set_train.ts_scaler.mean,
                                        scale_event_mean=data_set_train.event_scaler.mean,
                                        scale_event_std=data_set_train.event_scaler.std,sample_type=self.args.sample_type,type=self.args.data_type)


            data_loader_test = DataLoader(
                data_set_test,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=True)

            return data_set_train,data_set_valid,data_set_test,data_loader_train,data_loader_valid,data_loader_test

    def _select_optimizer(self):
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            return model_optim
        elif 'kae' in self.args.model:
            ts_optim = optim.Adam(self.model.ts_informer.parameters(),lr=self.args.learning_rate)
            event_optim = optim.Adam(self.model.event_informer.parameters(),lr=self.args.learning_rate)
            return ts_optim,event_optim
        elif 'csea' in self.args.model:
            ts_optim = optim.Adam(self.model.ts_informer.parameters(), lr=self.args.learning_rate)
            event_optim = optim.Adam(self.model.event_informer.parameters(), lr=self.args.learning_rate)
            out_optim = optim.Adam(self.model.final_output.parameters(),lr=self.args.learning_rate)
            return ts_optim, event_optim,out_optim
        else:
            ts_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            return ts_optim
    
    def _select_criterion(self,loss_type='mse',loss_beta=1.0):
        if loss_type == 'mse':
            print("loss type: l2 loss")
            criterion =  nn.MSELoss()
        elif loss_type == 'mae':
            print('loss type: l1 loss')
            criterion = nn.L1Loss()
        elif loss_type == 'huber':
            print("loss type: huber")
            criterion = nn.SmoothL1Loss(beta=loss_beta)
        else:
            print("loss type: l2 loss")
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            self.model.eval()
            total_loss = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                if self.args.sample_type > 3:
                    batch_y = batch_y.unsqueeze(2).contiguous()
                    batch_x = batch_x.unsqueeze(2).contiguous()

                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)
            total_loss = np.average(total_loss)
            self.model.train()
            return total_loss
        elif 'kae' in self.args.model:
            self.model.eval()

            total_vali_loss = []
            ar_base_loss = []
            time_series_loss = []
            ar_loss = []

            ts_loss = self._select_criterion(loss_type=self.args.loss, loss_beta=self.args.loss_beta)
            event_loss = self._select_criterion(loss_type=self.args.event_loss, loss_beta=self.args.loss_beta)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y, event_x_mark, event_y_mark) in enumerate(vali_loader):
                if self.args.sample_type > 3:
                    batch_y = batch_y.unsqueeze(2).contiguous()
                    batch_x = batch_x.unsqueeze(2).contiguous()
                pred_ts, true_ts = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, type=1)
                loss_ts_item = ts_loss(pred_ts, true_ts)
                time_series_loss.append(loss_ts_item.item())

                pre_ts_trans = vali_data.inverse_transform(pred_ts, type=1)

                pre_ts_trans = pre_ts_trans.transpose(1, 2).contiguous()

                pre_ts_trans = torch.flip(pre_ts_trans, dims=[1]).contiguous()

                pre_ts_trans = vali_data.event_scaler.transform(pre_ts_trans)

                pre_ts_trans_x = pre_ts_trans[:, (self.args.c_out - (
                            self.args.event_seq_len + self.args.event_pred_len)):(self.args.c_out - (
                    self.args.event_pred_len)), :]
                pre_ts_trans_y = pre_ts_trans[:, -(self.args.event_label_len + self.args.event_pred_len):, :]

                # self.model.event_informer.eval()
                pred_event, true_event = self._process_one_batch(
                    vali_data, event_x, event_y, event_x_mark, event_y_mark, type=2)
                ar_loss0 = event_loss(pred_event, true_event)

                ar_base_loss.append(ar_loss0.item())
                pred_pre_ts, true_pre_ts = self._process_one_batch(
                    vali_data, pre_ts_trans_x, pre_ts_trans_y, event_x_mark, event_y_mark, type=2
                )
                ar_loss1 = event_loss(pred_pre_ts, true_pre_ts)
                ar_loss.append(ar_loss1.item())

                if ar_loss1 > ar_loss0:
                    now_total_loss = loss_ts_item + self.args.event_alpha * (ar_loss1 - ar_loss0)
                else:
                    now_total_loss = loss_ts_item + self.args.event_alpha * (ar_loss0 - ar_loss1)

                total_vali_loss.append(now_total_loss.item())

            total_vali_loss_avg = np.average(total_vali_loss)
            ar_base_loss_avg = np.average(ar_base_loss)
            time_series_loss_avg = np.average(time_series_loss)
            ar_loss_avg = np.average(ar_loss)
            self.model.train()

            return total_vali_loss_avg,ar_base_loss_avg,time_series_loss_avg,ar_loss_avg
        elif 'csea' in self.args.model:
            self.model.eval()
            total_vali_loss = []
            # ar_base_loss = []
            time_series_loss = []
            ar_loss = []
            ts_loss = self._select_criterion(loss_type=self.args.loss, loss_beta=self.args.loss_beta)
            event_loss = self._select_criterion(loss_type=self.args.event_loss, loss_beta=self.args.loss_beta)
            out_loss = self._select_criterion(loss_type=self.args.out_loss, loss_beta=self.args.loss_beta)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y, event_x_mark, event_y_mark) in enumerate(
                    vali_loader):
                if self.args.sample_type > 3:
                    batch_y = batch_y.unsqueeze(2).contiguous()
                    batch_x = batch_x.unsqueeze(2).contiguous()

                pred_event, true_event = self._process_one_batch(
                    vali_data, event_x, event_y, event_x_mark, event_y_mark, type=2)
                loss_event_item = event_loss(pred_event, true_event)

                ar_loss.append(loss_event_item.item())

                pred_ts, true_ts = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, type=1)

                loss_ts_item = ts_loss(pred_ts, true_ts)
                time_series_loss.append(loss_ts_item.item())



                pred_ts_final, true_ts_final = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, type=1)

                pred_ts_final_input = pred_ts_final.clone().detach()
                pred_ts_final_predict = pred_ts_final_input[:, :, :self.args.event_pred_len]

                true_ts_final_predict = true_ts_final.clone().detach()[:, :, :self.args.event_pred_len]

                pred_event_final, true_event_final = self._process_one_batch(
                    vali_data, event_x, event_y, event_x_mark, event_y_mark, type=2)

                pred_event_final_input = pred_event_final.clone().detach()
                pred_event_final_input2 = vali_data.inverse_transform(pred_event_final_input, type=2)
                pred_event_final_input2 = pred_event_final_input2.transpose(1, 2).contiguous()
                pred_event_final_input2 = vali_data.ts_scaler.transform(pred_event_final_input2)
                pred_event_final_input2 = pred_event_final_input2.transpose(1, 2).contiguous()
                pred_event_final_predict = pred_event_final_input2.clone().detach()[:, :, :]

                pred_ts_final_predict = pred_ts_final_predict.transpose(1, 2).contiguous()
                # print("ts final input shape:")
                # print(pred_ts_final_predict.shape)
                # print("event final input shape:")
                # print(pred_event_final_predict.shape)

                # forward(self,x_enc, x_mark_enc, x_dec, x_mark_dec,
                #                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,type=1)
                final_predict_res = self.model.forward(x_enc=pred_ts_final_predict, x_mark_enc=None,
                                                       x_dec=pred_event_final_predict, x_mark_dec=None, type=3)

                if self.args.out_kernel_size < 0:
                    final_predict_res = final_predict_res.unsqueeze(2).contiguous()
                else:
                    final_predict_res = final_predict_res.transpose(1, 2).contiguous()
                # pred_ts, true_ts
                final_loss_item = out_loss(final_predict_res, true_ts_final_predict)
                total_vali_loss.append(final_loss_item.item())

            total_vali_loss_avg = np.average(total_vali_loss)
            ar_loss_avg = np.average(ar_loss)
            time_series_loss_avg = np.average(time_series_loss)
            self.model.train()

            return total_vali_loss_avg, time_series_loss_avg, ar_loss_avg

        else:
            self.model.eval()

            total_vali_loss = []
            # ar_base_loss = []
            # time_series_loss = []
            # ar_loss = []

            ts_loss = self._select_criterion(loss_type=self.args.loss, loss_beta=self.args.loss_beta)
            # event_loss = self._select_criterion(loss_type=self.args.event_loss, loss_beta=self.args.loss_beta)

            for i, (
            batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y, event_x_mark, event_y_mark) in enumerate(
                    vali_loader):

                if self.args.sample_type > 3:
                    batch_y = batch_y.unsqueeze(2).contiguous()
                    batch_x = batch_x.unsqueeze(2).contiguous()

                pred_ts, true_ts = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, type=1)
                loss_ts_item = ts_loss(pred_ts, true_ts)
                total_vali_loss.append(loss_ts_item.item())

            total_vali_loss_avg = np.average(total_vali_loss)

            self.model.train()
            # , ar_base_loss_avg, time_series_loss_avg, ar_loss_avg

            return total_vali_loss_avg






    def train(self, setting):
        # data_set_train,data_set_valid,data_set_test,data_loader_train,data_loader_valid,data_loader_test
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

            path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(path):
                os.makedirs(path)

            time_now = time.time()

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            model_optim = self._select_optimizer()
            criterion = self._select_criterion(loss_type=self.args.loss, loss_beta=self.args.loss_beta)

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1

                    if self.args.sample_type > 3:
                        batch_y = batch_y.unsqueeze(2).contiguous()
                        batch_x = batch_x.unsqueeze(2).contiguous()

                    model_optim.zero_grad()
                    pred, true = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(pred, true)
                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path,type=0)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

            return self.model
        elif 'kae' in self.args.model:
            train_data, valid_data,test_data,train_loader,vali_loader,test_loader = self._get_data(flag='train')
            path = os.path.join(os.path.join(os.path.join(self.args.checkpoints, self.args.model),self.args.app),setting)

            if not os.path.exists(path):
                os.makedirs(path)

            np.save(os.path.join(path, "ts_mean.npy"),train_data.ts_scaler.mean)
            np.save(os.path.join(path, "ts_std.npy"), train_data.ts_scaler.std)

            np.save(os.path.join(path, "event_mean.npy"), train_data.event_scaler.mean)
            np.save(os.path.join(path, "event_std.npy"), train_data.event_scaler.std)
            print("save the mean and std for the standard operation successfully!!!")
            time_now = time.time()

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            ts_optim,event_optim = self._select_optimizer()
            ts_loss = self._select_criterion(loss_type=self.args.loss, loss_beta=self.args.loss_beta)
            event_loss = self._select_criterion(loss_type=self.args.event_loss,loss_beta=self.args.loss_beta)

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                total_train_loss = []
                ar_base_loss = []
                time_series_loss = []
                ar_loss = []

                self.model.train()
                epoch_time = time.time()
                # seq_ts_x,seq_ts_y,seq_ts_x_mark,seq_ts_y_mark,seq_event_x,seq_event_y,seq_event_x_mark,seq_event_y_mark
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,event_x,event_y,event_x_mark,event_y_mark) in enumerate(train_loader):

                    iter_count += 1
                    ts_optim.zero_grad()
                    event_optim.zero_grad()

                    if self.args.sample_type > 3:
                        batch_y = batch_y.unsqueeze(2).contiguous()
                        batch_x = batch_x.unsqueeze(2).contiguous()

                    # print(event_x.shape)
                    # print(event_y.shape)
                    pred_event, true_event = self._process_one_batch(
                        train_data, event_x, event_y, event_x_mark, event_y_mark,type=2)
                    loss_event_item = event_loss(pred_event, true_event)

                    # print("pred event shape:")
                    # print(pred_event.shape)
                    #
                    # print("true event shape:")
                    # print(true_event.shape)


                    if self.args.use_amp:
                        scaler.scale(loss_event_item).backward()
                        scaler.step(event_optim)
                        scaler.update()
                    else:
                        loss_event_item.backward()
                        event_optim.step()

                    ts_optim.zero_grad()
                    event_optim.zero_grad()

                    pred_ts, true_ts = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, type=1)
                    loss_ts_item = ts_loss(pred_ts, true_ts)
                    time_series_loss.append(loss_ts_item.item())

                    # print("pred timeseries shape:")
                    # print(pred_ts.shape)
                    #
                    # print("true timeseries shape:")
                    # print(true_ts.shape)
                    pre_ts_trans = train_data.inverse_transform(pred_ts,type=1)

                    pre_ts_trans = pre_ts_trans.transpose(1,2).contiguous()

                    pre_ts_trans = torch.flip(pre_ts_trans,dims=[1]).contiguous()

                    # print("predict event for the ar process, shape:")
                    # print(pre_ts_trans.shape)

                    pre_ts_trans = train_data.event_scaler.transform(pre_ts_trans)
                    # self.args.pred_len
                    # self.args.pred_len

                    # print(self.args.c_out-(self.args.event_seq_len+self.args.event_pred_len))
                    # print(self.args.c_out-(self.args.event_pred_len))
                    pre_ts_trans_x = pre_ts_trans[:,self.args.c_out-(self.args.event_seq_len+self.args.event_pred_len):self.args.c_out-(self.args.event_pred_len),:]
                    pre_ts_trans_y = pre_ts_trans[:,-(self.args.event_label_len+self.args.event_pred_len):,:]

                    # self.model.event_informer.eval()

                    pred_event, true_event = self._process_one_batch(
                        train_data, event_x, event_y, event_x_mark, event_y_mark, type=2)
                    ar_loss0 = event_loss(pred_event, true_event)

                    # print("ar regularization: pred event shape:")
                    # print(pred_event.shape)
                    #
                    # print("ar regularization: true event shape:")
                    # print(true_event.shape)

                    ar_base_loss.append(ar_loss0.item())

                    # print("ar input x shape:")
                    # print(pre_ts_trans_x.shape)
                    # print("ar input y shape:")
                    # print(pre_ts_trans_y.shape)

                    pred_pre_ts,true_pre_ts = self._process_one_batch(
                        train_data,pre_ts_trans_x,pre_ts_trans_y,event_x_mark,event_y_mark,type=2
                    )

                    # print("ar regularization: ar output predict shape:")
                    # print(pred_pre_ts.shape)
                    # print("ar regularization: ar output true shape:")
                    # print(true_pre_ts.shape)

                    ar_loss1 = event_loss(pred_pre_ts,true_pre_ts)
                    ar_loss.append(ar_loss1.item())

                    if ar_loss1 > ar_loss0:
                        now_total_loss = loss_ts_item + self.args.event_alpha * (ar_loss1 - ar_loss0)
                    else:
                        now_total_loss = loss_ts_item + self.args.event_alpha * (ar_loss0 - ar_loss1)

                    total_train_loss.append(now_total_loss.item())

                    if self.args.use_amp:
                        scaler.scale(now_total_loss).backward()
                        scaler.step(ts_optim)
                        scaler.update()
                    else:
                        now_total_loss.backward()
                        ts_optim.step()

                    ts_optim.zero_grad()
                    event_optim.zero_grad()

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | total_loss: {2:.7f} ts_loss: {3:.7f} ar_base_loss: {4:.7f} ar_loss: {5:.7f}".format(i + 1, epoch + 1, total_train_loss[-1],time_series_loss[-1],ar_base_loss[-1],ar_loss[-1]))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                total_train_loss_avg = np.average(total_train_loss)
                ar_base_loss_avg = np.average(ar_base_loss)
                ar_loss_avg = np.average(ar_loss)
                time_series_loss_avg = np.average(time_series_loss)

                # total_vali_loss_avg,ar_base_loss_avg,time_series_loss_avg,ar_loss_avg
                vali_loss,vali_ar_base,vali_time_series_loss,vali_ar_loss = self.vali(vali_data=valid_data, vali_loader=vali_loader,criterion=ts_loss)
                # test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Total Loss: {2:.7f} Vali Total Loss: {3:.7f}".format(
                    epoch + 1, train_steps, total_train_loss_avg, vali_loss))
                print("Epoch: {0}, Steps: {1} | Train Time series Loss: {2:.7f} Vali Time series Loss: {3:.7f}".format(
                    epoch + 1, train_steps, time_series_loss_avg, vali_time_series_loss))
                print("Epoch: {0}, Steps: {1} | Train AR base Loss: {2:.7f} Vali AR base Loss: {3:.7f}".format(
                    epoch + 1, train_steps, ar_base_loss_avg, vali_ar_base))
                print("Epoch: {0}, Steps: {1} | Train AR Loss: {2:.7f} Vali AR Loss: {3:.7f}".format(
                    epoch + 1, train_steps, ar_loss_avg, vali_ar_loss))

                early_stopping(vali_time_series_loss, self.model, path,type=2)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(ts_optim, epoch + 1, self.args)
                adjust_learning_rate(event_optim,epoch+1,self.args)

        elif 'csea' in self.args.model:
            train_data, valid_data, test_data, train_loader, vali_loader, test_loader = self._get_data(flag='train')
            path = os.path.join(os.path.join(os.path.join(self.args.checkpoints, self.args.model), self.args.app),
                                setting)

            if not os.path.exists(path):
                os.makedirs(path)

            np.save(os.path.join(path, "ts_mean.npy"), train_data.ts_scaler.mean)
            np.save(os.path.join(path, "ts_std.npy"), train_data.ts_scaler.std)

            np.save(os.path.join(path, "event_mean.npy"), train_data.event_scaler.mean)
            np.save(os.path.join(path, "event_std.npy"), train_data.event_scaler.std)
            print("save the mean and std for the standard operation successfully!!!")
            time_now = time.time()

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            ts_optim, event_optim,out_optim = self._select_optimizer()
            ts_loss = self._select_criterion(loss_type=self.args.loss, loss_beta=self.args.loss_beta)
            event_loss = self._select_criterion(loss_type=self.args.event_loss, loss_beta=self.args.loss_beta)
            out_loss = self._select_criterion(loss_type=self.args.out_loss,loss_beta=self.args.loss_beta)

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                total_train_loss = []
                ar_loss = []
                time_series_loss = []
                # ar_loss = []

                self.model.train()
                epoch_time = time.time()
                # seq_ts_x,seq_ts_y,seq_ts_x_mark,seq_ts_y_mark,seq_event_x,seq_event_y,seq_event_x_mark,seq_event_y_mark
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y, event_x_mark,event_y_mark) in enumerate(train_loader):
                    if self.args.sample_type > 3:
                        batch_y = batch_y.unsqueeze(2).contiguous()
                        batch_x = batch_x.unsqueeze(2).contiguous()

                    iter_count += 1

                    ts_optim.zero_grad()
                    event_optim.zero_grad()
                    out_optim.zero_grad()

                    # print(event_x.shape)
                    # print(event_y.shape)
                    pred_event, true_event = self._process_one_batch(
                        train_data, event_x, event_y, event_x_mark, event_y_mark, type=2)
                    loss_event_item = event_loss(pred_event, true_event)

                    ar_loss.append(loss_event_item.item())

                    if self.args.use_amp:
                        scaler.scale(loss_event_item).backward()
                        scaler.step(event_optim)
                        scaler.update()
                    else:
                        loss_event_item.backward()
                        event_optim.step()

                    ts_optim.zero_grad()
                    event_optim.zero_grad()
                    out_optim.zero_grad()

                    pred_ts, true_ts = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, type=1)

                    loss_ts_item = ts_loss(pred_ts, true_ts)
                    time_series_loss.append(loss_ts_item.item())

                    if self.args.use_amp:
                        scaler.scale(loss_ts_item).backward()
                        scaler.step(ts_optim)
                        scaler.update()
                    else:
                        loss_ts_item.backward()
                        ts_optim.step()

                    pred_ts_final, true_ts_final = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, type=1)

                    pred_ts_final_input = pred_ts_final.clone().detach()
                    pred_ts_final_predict = pred_ts_final_input[:,:,:self.args.event_pred_len]

                    true_ts_final_predict = true_ts_final.clone().detach()[:,:,:self.args.event_pred_len]

                    pred_event_final, true_event_final = self._process_one_batch(
                        train_data, event_x, event_y, event_x_mark, event_y_mark, type=2)

                    pred_event_final_input = pred_event_final.clone().detach()
                    pred_event_final_input2 = train_data.inverse_transform(pred_event_final_input, type=2)
                    pred_event_final_input2 = pred_event_final_input2.transpose(1,2).contiguous()
                    pred_event_final_input2 = train_data.ts_scaler.transform(pred_event_final_input2)
                    pred_event_final_input2 = pred_event_final_input2.transpose(1, 2).contiguous()
                    pred_event_final_predict = pred_event_final_input2.clone().detach()[:, :, :]

                    pred_ts_final_predict = pred_ts_final_predict.transpose(1, 2).contiguous()
                    # print("ts final input shape:")
                    # print(pred_ts_final_predict.shape)
                    # print("event final input shape:")
                    # print(pred_event_final_predict.shape)
                    # forward(self,x_enc, x_mark_enc, x_dec, x_mark_dec,
                    #                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,type=1)
                    final_predict_res = self.model.forward(x_enc=pred_ts_final_predict,x_mark_enc=None,x_dec=pred_event_final_predict,x_mark_dec=None,type=3)

                    if self.args.out_kernel_size < 0:
                        final_predict_res = final_predict_res.unsqueeze(2).contiguous()
                    else:
                        final_predict_res = final_predict_res.transpose(1,2).contiguous()
                    # pred_ts, true_ts
                    final_loss_item = out_loss(final_predict_res,true_ts_final_predict)
                    total_train_loss.append(final_loss_item.item())

                    if self.args.use_amp:
                        scaler.scale(final_loss_item).backward()
                        scaler.step(out_optim)
                        scaler.update()
                    else:
                        final_loss_item.backward()
                        out_optim.step()

                    out_optim.zero_grad()
                    ts_optim.zero_grad()
                    event_optim.zero_grad()

                    if (i + 1) % 100 == 0:
                        print(
                            "\titers: {0}, epoch: {1} | total_loss: {2:.7f} |ts_loss: {3:.7f} |ar_loss: {4:.7f}".format(
                                i + 1, epoch + 1, total_train_loss[-1], time_series_loss[-1],
                                ar_loss[-1]))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                #
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                total_train_loss_avg = np.average(total_train_loss)
                # ar_base_loss_avg = np.average(ar_base_loss)
                ar_loss_avg = np.average(ar_loss)
                time_series_loss_avg = np.average(time_series_loss)

                vali_loss, vali_time_series_loss, vali_ar_loss = self.vali(vali_data=valid_data,
                                                                                         vali_loader=vali_loader,
                                                                                         criterion=ts_loss)

                #
                print("Epoch: {0}, Steps: {1} | Train Total Loss: {2:.7f} Vali Total Loss: {3:.7f}".format(
                    epoch + 1, train_steps, total_train_loss_avg, vali_loss))
                print("Epoch: {0}, Steps: {1} | Train Time series Loss: {2:.7f} Vali Time series Loss: {3:.7f}".format(
                    epoch + 1, train_steps, time_series_loss_avg, vali_time_series_loss))
                print("Epoch: {0}, Steps: {1} | Train AR Loss: {2:.7f} Vali AR Loss: {3:.7f}".format(
                    epoch + 1, train_steps, ar_loss_avg, vali_ar_loss))
                #
                early_stopping(vali_loss, self.model, path, type=4)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(ts_optim, epoch + 1, self.args)
                adjust_learning_rate(event_optim, epoch + 1, self.args)
                adjust_learning_rate(out_optim,epoch+1,self.args)

        else:
            train_data, valid_data, test_data, train_loader, vali_loader, test_loader = self._get_data(flag='train')
            path = os.path.join(os.path.join(os.path.join(self.args.checkpoints, self.args.model), self.args.app),
                                setting)

            if not os.path.exists(path):
                os.makedirs(path)

            np.save(os.path.join(path, "ts_mean.npy"), train_data.ts_scaler.mean)
            np.save(os.path.join(path, "ts_std.npy"), train_data.ts_scaler.std)

            np.save(os.path.join(path, "event_mean.npy"), train_data.event_scaler.mean)
            np.save(os.path.join(path, "event_std.npy"), train_data.event_scaler.std)
            print("save the mean and std for the standard operation successfully!!!")
            time_now = time.time()

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            # , event_optim
            ts_optim = self._select_optimizer()
            ts_loss = self._select_criterion(loss_type=self.args.loss, loss_beta=self.args.loss_beta)
            # event_loss = self._select_criterion(loss_type=self.args.event_loss, loss_beta=self.args.loss_beta)

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                total_train_loss = []
                ar_base_loss = []
                time_series_loss = []
                ar_loss = []

                self.model.train()
                epoch_time = time.time()
                # seq_ts_x,seq_ts_y,seq_ts_x_mark,seq_ts_y_mark,seq_event_x,seq_event_y,seq_event_x_mark,seq_event_y_mark
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y, event_x_mark,
                        event_y_mark) in enumerate(train_loader):

                    if self.args.sample_type > 3:
                        batch_y = batch_y.unsqueeze(2).contiguous()
                        batch_x = batch_x.unsqueeze(2).contiguous()

                    iter_count += 1
                    ts_optim.zero_grad()

                    pred_ts, true_ts = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, type=1)
                    loss_ts_item = ts_loss(pred_ts, true_ts)
                    # time_series_loss.append(loss_ts_item.item()
                    total_train_loss.append(loss_ts_item.item())

                    if self.args.use_amp:
                        scaler.scale(loss_ts_item).backward()
                        scaler.step(ts_optim)
                        scaler.update()
                    else:
                        loss_ts_item.backward()
                        ts_optim.step()

                    ts_optim.zero_grad()

                    if (i + 1) % 100 == 0:
                        print(
                            "\titers: {0}, epoch: {1} | total_loss: {2:.7f}".format(
                                i + 1, epoch + 1, total_train_loss[-1]))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                total_train_loss_avg = np.average(total_train_loss)


                # total_vali_loss_avg,ar_base_loss_avg,time_series_loss_avg,ar_loss_avg
                vali_loss = self.vali(vali_data=valid_data,vali_loader=vali_loader,criterion=ts_loss)
                # test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Total Loss: {2:.7f} Vali Total Loss: {3:.7f}".format(
                    epoch + 1, train_steps, total_train_loss_avg, vali_loss))

                early_stopping(vali_loss, self.model, path,type=0)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(ts_optim, epoch + 1, self.args)


    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        # # './results/'
        # parser.add_argument("--result_dir",type=str,default='./results/',help='result dir path')
        # # args.data_type
        # parser.add_argument("--data_type",type=int,default=0,help='data type')
        # './results/' + setting +'/'
        # folder_path =
        folder_path = os.path.join(os.path.join(os.path.join(self.args.result_dir, self.args.model), self.args.app),
                            setting)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        # folder_path+'metrics.npy'
        np.save(os.path.join(folder_path,"metric.npy"), np.array([mae, mse, rmse, mape, mspe]))
        # folder_path+'pred.npy'
        np.save(os.path.join(folder_path,'pred.npy'), preds)
        # folder_path+'true.npy'
        np.save(os.path.join(folder_path,'true.npy'), trues)

        return

    def predict_kae(self,setting,load=False):
        shuffle_flag = False;drop_last = False;batch_size = 1;freq = self.args.detail_freq

        path = os.path.join(os.path.join(os.path.join(self.args.checkpoints, self.args.model), self.args.app),
                            setting)
        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        ts_means = np.load(os.path.join(path,"ts_mean.npy"),allow_pickle=True)
        ts_stds = np.load(os.path.join(path,'ts_std.npy'),allow_pickle=True)
        event_means = np.load(os.path.join(path,"event_mean.npy"),allow_pickle=True)
        event_stds = np.load(os.path.join(path,"event_std.npy"),allow_pickle=True)


        timeenc = 0 if self.args.embed != 'timeF' else 1

        data_set_test = Dataset_QPS(root_path=self.args.root_path, flag='test',
                                    ts_size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
                                    event_size=[self.args.event_seq_len, self.args.event_label_len, self.args.event_pred_len],
                                    app_group=self.args.app, freq=freq, timeenc=timeenc,
                                    using_index=None, scale_type=2,
                                    scale_ts_std=ts_stds,
                                    scale_ts_mean=ts_means,
                                    scale_event_mean=event_means,
                                    scale_event_std=event_stds,sample_type=self.args.sample_type,type=self.args.data_type)


        data_loader_test = DataLoader(
            data_set_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False)

        if load:
            # path = os.path.join(self.args.checkpoints, setting)
            path = os.path.join(os.path.join(os.path.join(self.args.checkpoints, self.args.model), self.args.app),
                                setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []
        trues = []

        after_preds = []
        after_trues = []

        for i, (
        batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y, event_x_mark, event_y_mark) in enumerate(
            data_loader_test):

            if self.args.sample_type > 3:
                batch_y = batch_y.unsqueeze(2).contiguous()
                batch_x = batch_x.unsqueeze(2).contiguous()

            if 'csea' not in self.args.model:
                pred, true = self._process_one_batch(
                    data_set_test, batch_x, batch_y, batch_x_mark, batch_y_mark, type=1)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                # inverse_transform(self, data,type=1)
                after_pred = data_set_test.inverse_transform(pred)
                after_true = data_set_test.inverse_transform(true)

                after_preds.append(after_pred.detach().cpu().numpy())
                after_trues.append(after_true.detach().cpu().numpy())
            else:
                pred_ts_final, true_ts_final = self._process_one_batch(
                    data_set_test, batch_x, batch_y, batch_x_mark, batch_y_mark, type=1)

                pred_ts_final_input = pred_ts_final.clone().detach()
                pred_ts_final_predict = pred_ts_final_input[:, :, :self.args.event_pred_len]

                true_ts_final_predict = true_ts_final.clone().detach()[:, :, :self.args.event_pred_len]

                pred_event_final, true_event_final = self._process_one_batch(
                    data_set_test, event_x, event_y, event_x_mark, event_y_mark, type=2)

                pred_event_final_input = pred_event_final.clone().detach()
                pred_event_final_input2 = data_set_test.inverse_transform(pred_event_final_input, type=2)
                pred_event_final_input2 = pred_event_final_input2.transpose(1, 2).contiguous()
                pred_event_final_input2 = data_set_test.ts_scaler.transform(pred_event_final_input2)
                pred_event_final_input2 = pred_event_final_input2.transpose(1, 2).contiguous()
                pred_event_final_predict = pred_event_final_input2.clone().detach()[:, :, :]

                pred_ts_final_predict = pred_ts_final_predict.transpose(1, 2).contiguous()
                print("ts final input shape:")
                print(pred_ts_final_predict.shape)
                print("event final input shape:")
                print(pred_event_final_predict.shape)


                # forward(self,x_enc, x_mark_enc, x_dec, x_mark_dec,
                #                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,type=1)
                final_predict_res = self.model.forward(x_enc=pred_ts_final_predict, x_mark_enc=None,
                                                       x_dec=pred_event_final_predict, x_mark_dec=None, type=3)

                if self.args.out_kernel_size < 0:
                    final_predict_res = final_predict_res.unsqueeze(2).contiguous()
                else:
                    final_predict_res = final_predict_res.transpose(1, 2).contiguous()
                # pred_ts, true_ts

                preds.append(final_predict_res.detach().cpu().numpy())
                trues.append(true_ts_final_predict.detach().cpu().numpy())

                # inverse_transform(self, data,type=1)
                after_pred = data_set_test.inverse_transform(final_predict_res)
                after_true = data_set_test.inverse_transform(true_ts_final_predict)

                after_preds.append(after_pred.detach().cpu().numpy())
                after_trues.append(after_true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)

        after_preds = np.array(after_preds)
        after_trues = np.array(after_trues)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        print('test shape:', after_preds.shape, after_trues.shape)
        after_preds = after_preds.reshape(-1, after_preds.shape[-2], after_preds.shape[-1])
        after_trues = after_trues.reshape(-1, after_trues.shape[-2], after_trues.shape[-1])
        print('test shape:', after_preds.shape, after_trues.shape)

        # result save
        folder_path = os.path.join(
            os.path.join(os.path.join(self.args.result_dir, self.args.model), self.args.app),
            setting)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(os.path.join(folder_path,'metrics_before_inverse.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path,'pred_before_inverse.npy'), preds)
        np.save(os.path.join(folder_path,'true_before_inverse.npy'), trues)

        # result save
        # folder_path = './results/' + setting + '/'
        # './results/'
        folder_path = os.path.join(os.path.join(os.path.join(self.args.result_dir, self.args.model), self.args.app),
                            setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(after_preds, after_trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(os.path.join(folder_path,'metrics_after_inverse.npy'), np.array([mae, mse, rmse, mape, mspe]))

        np.save(os.path.join(folder_path,'real_prediction.npy'), after_preds)
        np.save(os.path.join(folder_path, 'real_groundtruth.npy'), after_trues)
        return



    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            if self.args.sample_type > 3:
                batch_y = batch_y.unsqueeze(2).contiguous()
                batch_x = batch_x.unsqueeze(2).contiguous()

            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = os.path.join(
            os.path.join(os.path.join(self.args.result_dir, self.args.model), self.args.app),
            setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,type=1):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)




        if self.args.model == 'informer' or self.args.model == 'informerstack':
            if self.args.padding==0:
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            elif self.args.padding==1:
                dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

            return outputs, batch_y

        elif 'kae' in self.args.model:
            if type >= 2:
                inp_length = self.args.event_pred_len
                inp_label_length = self.args.event_label_len
            else:
                inp_length = self.args.pred_len
                inp_label_length = self.args.label_len

            # print(inp_label_length)
            # print(inp_length)
            if self.args.padding == 0:
                dec_inp = torch.zeros([batch_y.shape[0], inp_length, batch_y.shape[-1]]).float()
            elif self.args.padding == 1:
                dec_inp = torch.ones([batch_y.shape[0], inp_length, batch_y.shape[-1]]).float()
            dec_inp = dec_inp.to(batch_y.device)
            dec_inp = torch.cat([batch_y[:, :inp_label_length, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark,type=type)[0]
                    else:
                        outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark,type=type)
            else:
                if self.args.output_attention:
                    outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark,type=type)[0]
                else:
                    outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark,type=type)
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)
            # inp_length
            # self.args.pred_len
            batch_y = batch_y[:, -inp_length:,:].to(self.device)

            return outputs, batch_y

        else:
            if type >= 2:
                inp_length = self.args.event_pred_len
                inp_label_length = self.args.event_label_len
            else:
                inp_length = self.args.pred_len
                inp_label_length = self.args.label_len

            # print(inp_label_length)
            # print(inp_length)
            if self.args.padding == 0:
                dec_inp = torch.zeros([batch_y.shape[0], inp_length, batch_y.shape[-1]]).float()
            elif self.args.padding == 1:
                dec_inp = torch.ones([batch_y.shape[0], inp_length, batch_y.shape[-1]]).float()

            dec_inp = dec_inp.to(batch_y.device)


            # print(dec_inp.shape)
            # print(batch_y.shape)
            # print(batch_x.shape)

            dec_inp = torch.cat([batch_y[:, :inp_label_length, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark,type=type)[0]
                    else:
                        outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark,type=type)
            else:
                if self.args.output_attention:
                    outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark,type=type)[0]
                else:
                    outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark,type=type)
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)
            # inp_length
            # self.args.pred_len
            batch_y = batch_y[:, -inp_length:,:].to(self.device)

            return outputs, batch_y

