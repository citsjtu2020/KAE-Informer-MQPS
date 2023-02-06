import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


def estimate_noise(data_input):
    return float(np.mean(np.abs(data_input[:-1] - data_input[1:])))

def getRandomIndex(n,x):
    # 索引范围为[0, n)，随机选x个不重复，注意replace=False才是不重复，replace=True则有可能重复
    # # 先根据上面的函数获取test_index
    # test_index = np.array(getRandomIndex(n, x))
    # # 再讲test_index从总的index中减去就得到了train_index
    # train_index = np.delete(np.arange(n), test_index)
    # ———————————————
    # 原文链接：https://blog.csdn.net/qq_32623363/article/details/104180152
    valid_index = np.random.choice(np.arange(n), size=x, replace=False)
    train_index = np.delete(np.arange(n),np.array(valid_index))
    return valid_index,train_index

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_QPS(Dataset):
    def __init__(self, root_path, flag='train', ts_size=None,event_size=None,
                 features='S', app_group='aggrehost',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None,using_index=None,scale_type=0,
                 scale_ts_mean=0.0,scale_ts_std=1.0,
                 scale_event_mean=0.0, scale_event_std=1.0,type=0,sample_type=0):

        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.type = type
        self.sample_type = sample_type
        print("input sample type is:")
        print(sample_type)
        print("sample type is: ")
        print(self.sample_type)

        self.ts_scaler = StandardScaler()
        self.event_scaler = StandardScaler()

        self.root_path = root_path
        self.app_group = app_group

        self.raw_using_index = using_index

        self.freq = freq
        self.timeenc = timeenc

        self.scale = scale
        self.est_noise = 1.0

        if ts_size == None:
            self.seq_len = 1440
            self.label_len = 120
            self.pred_len = 60
        else:
            self.seq_len = ts_size[0]
            self.label_len = ts_size[1]
            self.pred_len = ts_size[2]

        if event_size == None:
            self.event_seq_len = 7
            self.event_label_len = 2
            self.event_pred_len = 1

        else:
            self.event_seq_len = event_size[0]
            self.event_label_len = event_size[1]
            self.event_pred_len = event_size[2]

        print(self.event_seq_len)
        print(self.event_label_len)
        print(self.event_pred_len)

        self.scale_type = scale_type
        self.scale_ts_mean = scale_ts_mean
        self.scale_ts_std = scale_ts_std
        self.scale_event_mean = scale_event_mean
        self.scale_event_std = scale_event_std
        self.__read_data__()



    def __read_data__(self):
        assert self.flag in ['train', 'test', 'val']
        file_stat_ts = ""
        file_stat_event = ""
        if self.flag in ['train', 'val']:
            if self.type < 1:
                file_ts = "%s_train_ts.npy" % self.app_group
                file_event = "%s_train_event.npy" % self.app_group
            elif self.type < 3:
                file_ts = "%s_train_base_ts.npy" % self.app_group
                file_event = "%s_train_base_event.npy" % self.app_group
            elif self.type <7:
                file_ts = "%s_stat_train_ts.npy" % self.app_group
                file_event = "%s_stat_train_event.npy" % self.app_group

                file_stat_ts = "%s_stat_train_ts.npy" % self.app_group
                file_stat_event = "%s_stat_train_event.npy" % self.app_group
            else:
                file_ts = "%s_train_res_ts.npy" % self.app_group
                file_event = "%s_train_res_event.npy" % self.app_group

        else:
            if self.type < 1:
                file_ts = "%s_test_ts.npy" % self.app_group
                file_event = "%s_test_event.npy" % self.app_group
            elif self.type < 3:
                file_ts = "%s_test_base_ts.npy" % self.app_group
                file_event = "%s_test_base_event.npy" % self.app_group
            elif self.type < 7:
                file_ts = "%s_test_ts.npy" % self.app_group
                file_event = "%s_test_event.npy" % self.app_group

                file_stat_ts = "%s_stat_test_ts.npy" % self.app_group
                file_stat_event = "%s_stat_test_event.npy" % self.app_group
            else:
                file_ts = "%s_test_res_ts.npy" % self.app_group
                file_event = "%s_test_res_event.npy" % self.app_group

            # file_ts = "%s_test_ts.npy" % self.app_group
            # file_event = "%s_test_event.npy" % self.app_group


        data_ts = np.load(os.path.join(self.root_path, file_ts), allow_pickle=True)
        data_event = np.load(os.path.join(self.root_path, file_event), allow_pickle=True)

        if file_stat_event and file_stat_ts:
            data_stat_ts = np.load(os.path.join(self.root_path, file_stat_ts), allow_pickle=True)
            data_stat_event = np.load(os.path.join(self.root_path, file_stat_event), allow_pickle=True)

        data_event = np.flip(data_event,axis=1)

        if file_stat_ts and file_stat_event:
            data_stat_event = np.flip(data_stat_event,axis=1)


        data_ts_qps = data_ts[:, -(self.seq_len+self.pred_len):, 1:]
        data_event_qps = data_event[:, -(self.event_seq_len+self.event_pred_len):, 1:]

        if file_stat_ts and file_stat_event:
            data_stat_ts_qps = data_stat_ts[:, -(self.seq_len+self.pred_len):, 1:]
            data_stat_event_qps = data_stat_event[:, -(self.event_seq_len+self.event_pred_len):, 1:]

            print("need to add the stat data")
            data_ts_qps = data_ts_qps + data_stat_ts_qps
            data_event_qps = data_event_qps + data_stat_event_qps

        data_ts_date = data_ts[:,-(self.seq_len+self.pred_len):,0]
        ts_seq_len = data_ts_date.shape[1]
        data_event_date = data_event[:,-(self.event_seq_len+self.event_pred_len):,0]
        event_seq_len = data_event_date.shape[1]

        print(data_event_date.shape)

        data_ts_date_seq = np.concatenate([data_ts_date[i,:] for i in range(data_ts_date.shape[0])])
        print(data_ts_date_seq.shape)
        data_event_date_seq = np.concatenate([data_event_date[i, :] for i in range(data_event_date.shape[0])])

        data_ts_date_pdf = pd.DataFrame()
        data_ts_date_pdf['date'] = pd.Series(list(data_ts_date_seq))

        data_ts_date_pdf = data_ts_date_pdf.reset_index(drop=True)
        print(data_ts_date_pdf.head())


        data_event_date_pdf = pd.DataFrame()
        data_event_date_pdf['date'] = pd.Series(list(data_event_date_seq))
        data_event_date_pdf = data_event_date_pdf.reset_index(drop=True)
        print(data_event_date_seq.shape)
        print(data_event_date_pdf.head())


        data_ts_date_stmp = time_features(data_ts_date_pdf, timeenc=self.timeenc, freq=self.freq)
        data_event_date_stmp = time_features(data_event_date_pdf, timeenc=self.timeenc, freq=self.freq)

        data_ts_date_stmp = np.concatenate([np.expand_dims(data_ts_date_stmp[i:i + ts_seq_len, :], axis=0) for i in
                                            range(0, data_ts_date_stmp.shape[0], ts_seq_len)])


        data_event_date_stmp = np.concatenate([np.expand_dims(data_event_date_stmp[i:i + event_seq_len, :], axis=0) for i in
                                            range(0, data_event_date_stmp.shape[0], event_seq_len)])

        print(data_ts_date_stmp.shape)
        print(data_event_date_stmp.shape)
        print(data_ts_qps.shape)
        print(data_event_qps.shape)

        if self.scale:
            train_ts_data = data_ts_qps.astype(np.float32)
            train_event_data = data_event_qps.astype(np.float32)
            print(train_event_data.shape)
            print(train_ts_data.shape)
            self.ts_scaler.fit(train_ts_data,type=self.scale_type,input_mean=self.scale_ts_mean,input_std=self.scale_ts_std)
            self.event_scaler.fit(train_event_data,type=self.scale_type,input_mean=self.scale_event_mean,input_std=self.scale_event_std)

        print(self.ts_scaler.mean.shape)
        print(self.ts_scaler.std)
        print(self.event_scaler.mean.shape)
        print(self.event_scaler.std.shape)

        if self.raw_using_index is not None and len(self.raw_using_index) > 0:
            data_ts_date_stmp = data_ts_date_stmp[self.raw_using_index].astype(np.float)
            data_event_date_stmp = data_event_date_stmp[self.raw_using_index].astype(np.float)

            data_ts_qps = data_ts_qps[self.raw_using_index].astype(np.float)
            data_event_qps = data_event_qps[self.raw_using_index].astype(np.float)
        else:
            data_ts_date_stmp = data_ts_date_stmp.astype(np.float)
            data_event_date_stmp = data_event_date_stmp.astype(np.float)

            data_ts_qps = data_ts_qps.astype(np.float)
            data_event_qps = data_event_qps.astype(np.float)

        print("date shape:")
        print(data_ts_date_stmp.shape)
        print(data_event_date_stmp.shape)
        print("data shape:")
        print(data_ts_qps.shape)
        print(data_event_qps.shape)

        if self.scale:
            data_ts_qps = self.ts_scaler.transform(data_ts_qps)
            data_event_qps = self.event_scaler.transform(data_event_qps)

        data_ts_qps_base = data_ts_qps[:, self.seq_len, 0]
        self.est_noise = estimate_noise(data_ts_qps_base)

        self.ts_stamp = data_ts_date_stmp.astype(np.float)
        self.event_stamp = data_event_date_stmp.astype(np.float)
        self.ts_data = data_ts_qps.astype(np.float)
        self.event_data = data_event_qps.astype(np.float)

        if self.sample_type > 5:
            self.ts_data = self.ts_data[:,:,:1]


        print("date shape:")
        print(self.ts_stamp.shape)
        print(self.event_stamp.shape)
        print("data shape:")
        print(self.ts_data.shape)
        print(self.event_data.shape)

        # self.ts_data_x = data_ts_qps
        # self.ts_data_y = data_ts_qps
        # self.event_data_x = data_event_qps
        # self.event_data_y = data_event_qps

    def __getitem__(self, index):
        total_len = self.ts_data.shape[1]

        if self.sample_type > 3:
            seq_ts_x = self.ts_data[index, total_len - (self.seq_len + self.pred_len):total_len - self.pred_len, 0]
            seq_ts_y = self.ts_data[index, -(self.label_len + self.pred_len):, 0]
        elif self.sample_type > 5:
            seq_ts_x = self.ts_data[index, total_len - (self.seq_len + self.pred_len):total_len - self.pred_len, :1]
            seq_ts_y = self.ts_data[index, -(self.label_len + self.pred_len):, :1]
        else:
            seq_ts_x = self.ts_data[index,total_len-(self.seq_len+self.pred_len):total_len-self.pred_len,:]
            seq_ts_y = self.ts_data[index,-(self.label_len+self.pred_len):,:]

        total_event_len = self.event_data.shape[1]
        # :self.event_seq_len
        seq_event_x = self.event_data[index,total_event_len-(self.event_seq_len+self.event_pred_len):total_event_len-self.event_pred_len,:]
        seq_event_y = self.event_data[index,-(self.event_label_len+self.event_pred_len):,:]

        seq_ts_x_mark = self.ts_stamp[index,total_len-(self.seq_len+self.pred_len):total_len-self.pred_len,:]
        seq_ts_y_mark = self.ts_stamp[index,-(self.label_len+self.pred_len):,:]

        seq_event_x_mark = self.event_stamp[index,total_event_len-(self.event_seq_len+self.event_pred_len):total_event_len-self.event_pred_len,:]
        seq_event_y_mark = self.event_stamp[index,-(self.event_label_len+self.event_pred_len):,:]



        return seq_ts_x,seq_ts_y,seq_ts_x_mark,seq_ts_y_mark,seq_event_x,seq_event_y,seq_event_x_mark,seq_event_y_mark


    def __len__(self):
        return self.ts_data.shape[0]

    def inverse_transform(self, data,type=1):
        if type <= 1:
            return self.ts_scaler.inverse_transform(data)
        else:
            return self.event_scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        # self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()


        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
