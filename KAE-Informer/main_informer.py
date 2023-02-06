import argparse
import os
import torch
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE

from exp.exp_informer import Exp_Informer
def aggre_data(x_pdf):
    out_data = {}
    x_pdf_x = x_pdf.copy().sort_index().reset_index(drop=True)
    for c in x_pdf.columns:
        if c == 'ds':
            out_data[c] = [x_pdf[c].unique().tolist()[0]]
        else:
            out_data[c] = [x_pdf_x[c][0]]

    return pd.DataFrame(out_data)

import json
def save_config(config,filename):
    config_content = {}
    for key,value in config.items():
        # if key != 'job' and key != 'ns':
        config_content[key] = value
        # task_content['task_id'] = tasks['task_id']
    fw = open(filename, 'w', encoding='utf-8')
    dic_json = json.dumps(config_content, ensure_ascii=False, indent=4)
    fw.write(dic_json)
    fw.close()

def load_config(config_file):

    f = open(config_file,encoding='utf-8')
    res = f.read()
    config_content = json.loads(res)
    return config_content

def predict_long_short_acf(prop_data,y_data,predicted_smooth_data,true_smooth_data,win_predict=60, win_update=240,win_stat=240,
                           win_prop=1440, filter_step=3, lowpass_upper=24, short_gain=2, sample_rate=1440, min_cycle=7,
                           alpha=0.5, adjust=True, fft_rate=0.5, acf_type='gas', acf_alpha=1):
    data_space_len = predicted_smooth_data.shape[0]
    total_results = []
    # (1380, 60, 8)
    time01 = time.time()
    out_predicted = pd.DataFrame()
    for i in range(data_space_len):
        tmp_prop_predicted = prop_data[i,:,:]
        tmp_prop_predicted = tmp_prop_predicted[-60:,:]

        tmp_predicted = pd.DataFrame()
        tmp_predicted['ds'] = pd.Series(list(tmp_prop_predicted[:,0]))
        tmp_predicted['prop_predict'] = pd.Series(list(tmp_prop_predicted[:,1]))

        tmp_predicted = tmp_predicted.reset_index(drop=True)

        tmp_y_data = y_data[i,:,:].squeeze()
        tmp_y_data = tmp_y_data[-60:,:]
        tmp_predicted['y_data'] = pd.Series(list(tmp_y_data[:,1]))
        tmp_predicted = tmp_predicted.reset_index(drop=True)

        tmp_predicted_smooth = predicted_smooth_data[i,:,:]
        tmp_predicted['smooth_predict'] = pd.Series(list(tmp_predicted_smooth[:,0]))

        tmp_true_smooth = true_smooth_data[i,:,:]
        tmp_predicted['smooth_true'] = pd.Series(list(tmp_true_smooth[:,0]))




        tmp_predicted.reset_index(drop=True)
        tmp_predicted['y_predicted'] = tmp_predicted['smooth_predict'] + tmp_predicted['prop_predict']

        tmp_predicted.reset_index(drop=True)

        if i == 0:
            out_predicted = tmp_predicted.copy().reset_index(drop=True)
        else:
            out_predicted = pd.concat([out_predicted,tmp_predicted],axis=0).reset_index(drop=True)


    time02 = time.time()
    return out_predicted, time02 - time01

def compute_mape2(y_pred, y_true,mean_value):
    # [np.abs((y_pred[i] - y_true[i]) / np.abs(y_true[i])) for i in range(len(y_pred))]
    print(mean_value)
    res = (np.mean([np.abs((y_pred[i] - y_true[i])) for i in range(len(y_pred))]) / mean_value)
    print(res)
    return res

def compute_stpe(y_pred,y_true):
    err_res = np.sum([np.abs((y_pred[i] - y_true[i])) for i in range(len(y_pred))])
    total_res = np.sum([np.abs(y_pred[i]) for i in range(len(y_pred))]) + np.sum([np.abs(y_true[i]) for i in range(len(y_true))])

    print(err_res)
    print(total_res)
    return err_res / total_res

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

# , required=True
parser.add_argument('--model', type=str, default='kaeinformer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

# , required=True
parser.add_argument('--data', type=str, default='qps', help='data')
parser.add_argument("--data_item_size",type=int,default=480,help='data length')
# './results/'
parser.add_argument("--result_dir",type=str,default='./results/',help='result dir path')
# args.data_type
parser.add_argument('--exp_results',type=str,default='./exp_results/',help='dir path to save the predicted metric')
parser.add_argument("--data_type",type=int,default=0,help='data type')

parser.add_argument('--root_path', type=str, default='./data/data_train_test/', help='root path of the data file')
parser.add_argument("--app",type=str,default='app8-example',help='app group of the prediction')
parser.add_argument("--app_group",type=str,default='app8-example',help='app group of the prediction')

parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./KAE/checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=1440, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=120, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=60, help='prediction sequence length')

parser.add_argument('--sample_type',type=int,default=0)

parser.add_argument('--event_seq_len', type=int, default=7, help='input sequence length of Informer encoder')
parser.add_argument('--event_label_len', type=int, default=2, help='start token length of Informer decoder')
parser.add_argument('--event_pred_len', type=int, default=1, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
parser.add_argument('--c_out', type=int, default=8, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--event_d_model', type=int, default=384, help='dimension of event model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--event_n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--event_e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--event_d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--event_d_ff', type=int, default=1536, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--event_factor', type=int, default=3, help='probsparse attn factor for event informer')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument("--event_distil",default=False,help='whether to use distilling in encoder, using this argument means not using distilling')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--event_attn', type=str, default='full', help='attention used in encoder, options:[prob, full]')
parser.add_argument("--setting",type=str,default='',help="the setting to load")

parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--event_activation', type=str, default='gelu',help='event activation')

parser.add_argument("--prop_data",type=str, default='app8-example_test_prophet_ts.npy')
parser.add_argument("--y_data",type=str, default='app8-example_test_base_ts.npy')

parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.00024, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='huber',help='loss function')
parser.add_argument('--event_loss', type=str, default='mse',help='loss function')
parser.add_argument("--event_alpha",type=float,default=0.1,help='ar regularization')

parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument("--qvk_kernel_size",type=int,default=5,help='kernel size for qvk projection')
parser.add_argument("--event_qvk_kernel_size",type=int,default=3,help='kernel size for event qvk projection')

# out_kernel_size=5,out_hidden_size=256,
parser.add_argument("--out_kernel_size",type=int,default=3,help='kernel size for out projection')
parser.add_argument("--out_loss",type=str,default='huber',help='loss type of out projection')
parser.add_argument("--out_hidden_size",type=int,default=32,help='hidden size for out projection')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
parser.add_argument("--loss_beta",type=float,default=1.0,help="huber loss beta")
parser.add_argument("--scale_type",type=int,default=0,help="how to scale the input data")


args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer


aims = {}
aims[args.gpu] = [args.app]
final_results = {}
detail_results = {}


base_root_path = args.root_path
for item in aims[args.gpu]:
    final_results[item] = {}
    detail_results[item] = {}
    detail_results[item]['ts_extractor'] = {}
    final_results[item]['ts_extractor'] = {}
    detail_results[item]['kae-informer'] = {}
    final_results[item]['kae-informer'] = {}
    for ii in range(args.itr):
        # setting record of experiments
        args.app = item
        args.app_group = args.app

        args.root_path = os.path.join(base_root_path,item)
        print("now process app group: %s" % args.app)
        if 'csea' in args.model:
            setting = '{}_data_type_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_edm{}_nh{}_el{}_dl{}_df{}_edf{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}_loss_{}_beta_{}_alpha{}_qvk_k_{}_event_qvk_k_{}_out_k_{}_out_loss_{}_out_hidden_{}_{}'.format(
                args.model, args.data_type, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.event_d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                args.event_d_ff,
                args.attn, args.factor,
                args.embed, args.distil, args.mix, args.des, ii, args.loss, args.loss_beta, args.event_alpha,
                args.qvk_kernel_size, args.event_qvk_kernel_size, args.out_kernel_size, args.out_loss,
                args.out_hidden_size, int(time.time() / 3600))
        else:
            setting = '{}_data_type_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_edm{}_nh{}_el{}_dl{}_df{}_edf{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}_loss_{}_beta_{}_alpha{}_qvk_k_{}_event_qvk_k_{}_sample_type_{}_{}'.format(
                args.model, args.data_type, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.event_d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                args.event_d_ff,
                args.attn, args.factor,
                args.embed, args.distil, args.mix, args.des, ii, args.loss, args.loss_beta, args.event_alpha,
                args.qvk_kernel_size, args.event_qvk_kernel_size, args.sample_type,int(time.time() / 3600))

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        if args.do_predict:
            if args.setting:
                setting = args.setting
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict_kae(setting=setting, load=True)
            # os.path.join(args.root_path, args.app)
            args.prop_data = os.path.join(args.root_path, "%s_test_prophet_ts.npy" % args.app)
            prop_data = np.load(args.prop_data, allow_pickle=True)
            print(prop_data.shape)
            # os.path.join(args.root_path, args.app)
            args.y_data = os.path.join(args.root_path, "%s_test_base_ts.npy" % args.app)
            y_data = np.load(args.y_data, allow_pickle=True)
            print(y_data.shape)



            predicted_smooth_data = np.load(os.path.join(os.path.join(
                os.path.join(os.path.join(args.result_dir, args.model), args.app),
                setting), "real_prediction.npy"), allow_pickle=True)
            print(predicted_smooth_data.shape)
            true_smooth_data = np.load(os.path.join(os.path.join(
                os.path.join(os.path.join(args.result_dir, args.model), args.app),
                setting), "real_groundtruth.npy"), allow_pickle=True)

            if not os.path.exists(args.exp_results):
                os.makedirs(args.exp_results,exist_ok=True)
            save_dir = os.path.join(args.exp_results,args.app)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            total_results,overhead = predict_long_short_acf(prop_data=prop_data,y_data=y_data,predicted_smooth_data=predicted_smooth_data,true_smooth_data=true_smooth_data)
            aggre_results = total_results.groupby("ds").apply(aggre_data)
            total_results.to_csv(os.path.join(save_dir,"detail_results.csv"),index=None)
            aggre_results.to_csv(os.path.join(save_dir,"aggre_results.csv"),index=None)

            prop_mse = MSE(y_true=(total_results.y_data - total_results.copy().y_data.mean())/total_results.copy().y_data.std(), y_pred=(total_results.prop_predict-total_results.copy().y_data.mean())/total_results.copy().y_data.std())
            y_predict_mse = MSE(y_true=(total_results.y_data-total_results.copy().y_data.mean())/total_results.copy().y_data.std(), y_pred=(total_results.y_predicted-total_results.copy().y_data.mean())/total_results.copy().y_data.std())

            prop_aggre_mse = MSE(y_true=(aggre_results.y_data-aggre_results.copy().y_data.mean())/aggre_results.copy().y_data.std(), y_pred=(aggre_results.prop_predict-aggre_results.copy().y_data.mean())/aggre_results.copy().y_data.std())
            y_predict_aggre_mse = MSE(y_true=(aggre_results.y_data-aggre_results.copy().y_data.mean())/aggre_results.copy().y_data.std(), y_pred=(aggre_results.y_predicted-aggre_results.copy().y_data.mean())/aggre_results.copy().y_data.std())


            detail_results[item]['ts_extractor']['mse'] = prop_mse
            detail_results[item]['kae-informer']['mse'] = y_predict_mse

            final_results[item]['ts_extractor']['mse'] = prop_aggre_mse

            final_results[item]['kae-informer']['mse'] = y_predict_aggre_mse

            detail_results[item]['ts_extractor']['mape'] = (
                compute_mape2(y_true=total_results.y_data, y_pred=total_results.prop_predict, mean_value=total_results.y_data.mean()))
            detail_results[item]['kae-informer']['mape'] = (compute_mape2(y_true=total_results.y_data, y_pred=total_results.y_predicted,
                                                              mean_value=total_results.y_data.mean()))

            final_results[item]['ts_extractor']['mape'] = (
                compute_mape2(y_true=aggre_results.y_data, y_pred=aggre_results.prop_predict,
                              mean_value=aggre_results.y_data.mean()))

            final_results[item]['kae-informer']['mape'] = (
                compute_mape2(y_true=aggre_results.y_data, y_pred=aggre_results.y_predicted,
                              mean_value=aggre_results.y_data.mean()))

            final_results[item]['ts_extractor']['stpe'] = np.mean(
                compute_stpe(y_true=aggre_results.y_data, y_pred=aggre_results.prop_predict))
            final_results[item]['kae-informer']['stpe'] = np.mean(
                compute_stpe(y_true=aggre_results.y_data, y_pred=aggre_results.y_predicted))

            detail_results[item]['ts_extractor']['stpe'] = np.mean(
                compute_stpe(y_true=total_results.y_data, y_pred=total_results.prop_predict))

            detail_results[item]['kae-informer']['stpe'] = np.mean(
                compute_stpe(y_true=total_results.y_data, y_pred=total_results.y_predicted))

            print("detail metrics:")
            print(detail_results)

            print("aggregated results:")
            print(final_results)

            save_config(detail_results,os.path.join(save_dir, "detail_mse_mape_stpe_results.json"))
            save_config(final_results, os.path.join(save_dir, "aggre_mse_mape_stpe_results.json"))

    torch.cuda.empty_cache()

