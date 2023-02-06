import prophet
import argparse
from prophet import Prophet
import pandas as pd
# import os
import scipy
from scipy import signal
import os
# from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_squared_error as MSE
import math
# import neuralprophet
import dtaidistance
import time
from scipy import fftpack
from scipy.fftpack import fft, ifft
# import time

# import dtw
import numpy as np

import statsmodels
from statsmodels.tsa.stattools import acf, pacf
# ,plot_acf, plot_pacf
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.arima_model import ARIMA


def multi_diff(data, times=1):
    input_data = data.copy()
    for i in range(times):
        input_data = input_data.diff()
        input_data = input_data.dropna().reset_index(drop=True)

    return input_data.dropna().reset_index(drop=True)


# %%

def test_stationarity(timeseries):
    temp = np.array(timeseries)
    t = adfuller(temp)  # ADF检验
    output = pd.DataFrame(
        index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used", "Critical Value(1%)",
               "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    return output


# %%

# test_stationarity(time_data0.y_res.to_numpy())
def find_diff_time(data, column='y_res', diff_up_limit=4):
    input_data = data.copy()
    tmp_input_data = input_data.copy()
    #     if diff_up_limit == 0
    ts_test_res = test_stationarity(tmp_input_data[column].to_numpy())
    print("diff is %d, p-value=%f" % (0, ts_test_res['value']['p-value']))
    if ts_test_res['value']['p-value'] <= 0.05:
        return 0
    else:
        diff_times = 0
        for i in range(diff_up_limit):
            now_diff = i + 1
            tmp_input_data = input_data.copy()
            tmp_diff_res = multi_diff(tmp_input_data[[column]], times=now_diff)
            ts_test_res = test_stationarity(tmp_diff_res[column].to_numpy())
            if ts_test_res['value']['p-value'] <= 0.05:
                diff_times = now_diff
                print("diff is %d, p-value=%f" % (now_diff, ts_test_res['value']['p-value']))
                break
        if diff_times <= 0:
            diff_times = diff_up_limit
            print("diff is %d, p-value=%f" % (now_diff, ts_test_res['value']['p-value']))
        return diff_times

    # %%


def select_p_q_for_ar_ma(data, p_max, q_max, selected_d, column='y_res', p_inter=1, q_inter=1, mode=1):
    aic_matrix_res = []
    df0 = data.copy()
    lag_acf0 = list(acf(df0[column], nlags=np.min([1440, (df0.shape[0] / 2 - 1)])))
    lag_pacf0 = list(pacf(df0[column], nlags=np.min([1440, (df0.shape[0] / 2 - 1)])))
    pidx = 0
    qidx = 0
    for i in range(len(lag_acf0)):
        if lag_acf0[i] < 0.1:
            ends = True
            #             if i < q_max:
            #                 for j in range(i,q_max):
            #                     if lag_acf0[j] > 0.1:
            #                         ends = False
            #                         break
            if ends:
                qidx = i - 1
                if i == 0:
                    qidx = 0
                break
    for i in range(len(lag_pacf0)):
        if lag_pacf0[i] < 0.03:
            ends = True
            #             if i < p_max:
            #                 for j in range(i,p_max):
            #                     if lag_pacf0[j] > 0.1:
            #                         ends = False
            #                         break
            if ends:
                pidx = i - 1
                if i == 0:
                    pidx = 1
                break
    #     print(lag_acf0)
    #     print(lag_pacf0)
    print("pacf compute p for AR: %d, acf compute q for MA: %d" % (pidx, qidx))
    if p_max > 0:
        p_max0 = np.min([pidx, p_max])
    else:
        p_max0 = pidx
    if q_max > 0:
        q_max0 = np.min([qidx, q_max])
    else:
        q_max0 = qidx
    if mode > 0:
        print("selected p for AR: %d, selected q for MA: %d" % (p_max0, q_max0))
        return p_max0, q_max0
    else:
        for p in range(0, p_max + 1, p_inter):
            temp = []
            for q in range(0, q_max + 1, q_inter):
                print(p, q)
                try:
                    temp.append(ARIMA(df0[column].to_numpy(), (p, selected_d, q)).fit().aic)
                except Exception as e:
                    temp.append(None)
                    print(e)
            aic_matrix_res.append(temp)
        aic_matrix_res = pd.DataFrame(aic_matrix_res)
        p, q = aic_matrix_res.stack().idxmin()
        print(aic_matrix_res.shape)
        print(aic_matrix_res)
        print("selected p for AR: %d, selected q for MA: %d" % (p * p_inter, q * q_inter))
        return p * p_inter, q * q_inter


# %%

def arima_p_d_q_selection(data, p_max, q_max, d_max, column='y_res', p_inter=1, q_inter=1, mode=1):
    raw_data0 = data.copy()
    diff0 = find_diff_time(raw_data0, column=column, diff_up_limit=d_max)
    print("selected d for difference in arima: %d" % diff0)
    p, q = select_p_q_for_ar_ma(data, p_max, q_max, selected_d=diff0, column=column, p_inter=p_inter, q_inter=q_inter,
                                mode=mode)
    return p, diff0, q


# %%

def compute_mape(y_pred, y_true):
    res = [np.abs(y_pred[i] - y_true[i]) / np.abs(y_true[i]) for i in range(len(y_pred))]
    return res


# %%

def down_sample(x_pdf, column, window=10, way='median', alpha=0.6, adjust=False):
    input_data = x_pdf.copy().reset_index(drop=True)
    input_data = input_data.sort_values(['ds']).reset_index(drop=True)
    input_index = input_data.index
    out_qps = []
    out_ds = []
    print(len(input_index))
    print(window)
    #     d = [i for i in range(len(input_index)-1,0,-1*window)]
    #     print(d)
    for j in range(window - 1, len(input_index), 1 * window):
        #         print(j)
        down_d = mean_sample_predict(input_data[input_index[j - (window - 1)]:input_index[j] + 1], column,
                                     window=window, way=way, alpha=alpha, adjust=adjust)
        down_ds = input_data[input_index[j - (window - 1)]:input_index[j] + 1].ds.max()
        #         print(down_d)
        out_qps.append(down_d)
        out_ds.append(down_ds)

    out_pdf = pd.DataFrame({'ds': out_ds, column: out_qps})
    return out_pdf


# %%

def mean_sample_predict(x_pdf, column, window=10, way='median', alpha=0.6, adjust=False):
    input_data = x_pdf.copy()
    input_data = input_data.tail(window).sort_values(['ds']).reset_index(drop=True)
    #     print("predict")
    if way == 'median':
        return input_data[column].quantile(0.5)
    elif way == 'last':
        input_index = list(input_data.index)
        return input_data[column][input_index[-1]]
    elif way == 'first':
        input_index = list(input_data.index)
        return input_data[column][input_index[0]]
    elif way == 'mean':
        return input_data[column].mean()
    elif way == 'span':
        ewm_res = input_data[column].ewm(span=window).mean()
        ewm_index = list(ewm_res.index)
        return ewm_res[ewm_index[-1]]
    elif way == 'com':
        ewm_res = input_data[column].ewm(com=window).mean()
        ewm_index = list(ewm_res.index)
        return ewm_res[ewm_index[-1]]
    elif way == 'max':
        return input_data[column].max()
    elif way == 'min':
        return input_data[column].min()
    else:
        ewm_res = input_data[column].ewm(alpha=alpha, adjust=adjust).mean()
        ewm_index = list(ewm_res.index)
        return ewm_res[ewm_index[-1]]


# %%

# import numpy as np



# %%

def FFT(x_pdf, fs, column):
    X = x_pdf[[column]].dropna().copy()
    #     X = (X[column] - X[column].mean()) / X[column].std()
    X = X[column]
    L = X.shape[0]
    X = X.to_list()
    Y = fft(X)
    p2 = np.abs(Y)  # 双侧频谱
    p1 = p2[:int(L / 2)]

    f = np.arange(int((L / 2))) * fs / (L)
    p20 = list(2 * p1 / (L))
    p20[0] = p20[0] / 2
    return f, p20, Y


# %%

def generalize_FFT(x_pdf, fs, column):
    X = x_pdf[[column]].dropna().copy()
    X = (X[column] - X[column].mean()) / X[column].std()
    #     X = X[column]
    L = X.shape[0]
    X = X.to_list()
    Y = fft(X)
    p2 = np.abs(Y)  # 双侧频谱
    p1 = p2[:int(L / 2)]

    f = np.arange(int((L / 2))) * fs / (L)
    p20 = list(2 * p1 / (L))
    p20[0] = p20[0] / 2
    return f, p20, Y


# %%

def ACF(x_pdf, column, fs=1440, lower_limit_fp=12):
    lag_len = int(fs / lower_limit_fp)
    lag_len = np.min([int(x_pdf.shape[0] / 2), lag_len])
    lag_acf_once = acf(x_pdf[column], nlags=lag_len)
    lag_f = [fs / (i + 1) for i in range(1, len(lag_acf_once))]
    lag_acf_dict = {'freq': [], 'acf': []}
    for j in range(len(lag_f)):
        lag_acf_dict['freq'].append(lag_f[j])
        lag_acf_dict['acf'].append(lag_acf_once[j + 1])

    lag_acf_pdf = pd.DataFrame(lag_acf_dict)
    #     lag_acf_dict = {}
    #     for j in range(len(lag_f)):
    #         lag_acf_dict[lag_f[j]] = lag_acf_once[j+1]
    return lag_acf_pdf


# %%

def merge_fft_acf_res(fft_pdf, acf_pdf):
    input_fft = fft_pdf.copy().reset_index(drop=True)
    input_acf = acf_pdf.copy().reset_index(drop=True)

    fft_index = input_fft.index
    aim_acf_res = {'acf_freq': [], 'acf': [], 'freq': []}
    for j in fft_index:
        tmp_freq = input_fft['freq'][j]
        input_acf['res_freq'] = input_acf['freq'] - tmp_freq
        input_acf['res_freq'] = input_acf['res_freq'].abs()
        input_acf = input_acf.sort_values(['res_freq']).reset_index(drop=True)
        aim_res_freq = input_acf['res_freq'][0]
        aim_acf_res['acf_freq'].append(input_acf['freq'][0])
        aim_acf_res['freq'].append(tmp_freq)
        aim_acf_res['acf'].append(input_acf['acf'][0])

    aim_acf_res_pdf = pd.DataFrame(aim_acf_res)
    merge_fft = pd.merge(input_fft, aim_acf_res_pdf, left_on='freq', right_on='freq', how='inner')
    return merge_fft


# %%

def generate_sin_signal(A, freq, phase, t):
    S = A * math.sin(2 * math.pi * freq * t + phase)
    #     print(math.sin(2*math.pi*freq*t+phase))
    #     print(2*math.pi*freq*t+phase)
    return S


# %%

def compute_sin_signal(A, freq, phase, ts=[]):
    result = []
    for i in range(len(ts)):
        #         print(ts[i])
        S_tmp = generate_sin_signal(A, freq, phase, ts[i])
        result.append(S_tmp)
    return result


#         return S_tmp

# %%

def predict_short_stat(x_pdf, column, fs=1440, lower_limit_fp=12, fft_rate=0.5, acf_type='median', alpha=1,
                       win_predict=10):
    input_data = x_pdf.copy()
    f_list, p_list, Y_list = FFT(input_data, fs=fs, column=column)
    acf_pdf = ACF(x_pdf, column, fs=fs, lower_limit_fp=lower_limit_fp)
    #     lag_acf
    fft_f_power_phase = {'freq': [], 'power': [], 'phase': []}
    #     fft_f_phase = {}
    # #     fft
    for j in range(1, len(p_list)):
        fft_f_power_phase['freq'].append(f_list[j])
        fft_f_power_phase['power'].append(p_list[j])
        fft_f_power_phase['phase'].append(math.atan2(Y_list[j].real, Y_list[j].imag))

    fft_f_pdf = pd.DataFrame(fft_f_power_phase)

    if fft_rate >= 1:
        aim_fft_power = fft_f_pdf.power.quantile(0.99)
    elif fft_rate <= 0:
        aim_fft_power = fft_f_pdf.power.quantile(0.05)
    else:
        aim_fft_power = fft_f_pdf.power.quantile(fft_rate)

    aim_fft_f_pdf = fft_f_pdf.query("power >= %f" % aim_fft_power)
    aim_fft_f_pdf = aim_fft_f_pdf.reset_index(drop=True)
    #     fft_freq_dur = fs / 2 / len(p_list)
    out_merge_df = merge_fft_acf_res(aim_fft_f_pdf, acf_pdf)

    acf_mean = acf_pdf.acf.mean()
    acf_std = acf_pdf.acf.std()
    print(acf_pdf.acf.quantile(0.99))
    if acf_type == 'median':
        aim_acf_sc = acf_pdf.acf.median()
        out_merge_df = out_merge_df.query("acf>=%f" % aim_acf_sc).reset_index(drop=True)
    elif acf_type == 'mean':
        aim_acf_sc = acf_pdf.acf.mean()
        out_merge_df = out_merge_df.query("acf>=%f" % aim_acf_sc).reset_index(drop=True)
    elif acf_type == 'quantile':
        if alpha >= 1:
            aim_acf_sc = acf_pdf.acf.quantile(0.99)
        elif alpha <= 0:
            aim_acf_sc = acf_pdf.acf.quantile(0.05)
        else:
            aim_acf_sc = acf_pdf.acf.quantile(fft_rate)

        out_merge_df = out_merge_df.query("acf>=%f" % aim_acf_sc).reset_index(drop=True)
    else:
        if alpha < 0:
            alpha = 0
        aim_acf_sc_up = acf_mean + alpha * acf_std
        #         print(acf_pdf)
        tmp_acf_up = acf_pdf.acf.quantile(0.99)
        #         print(tmp_acf_up)
        #         print(aim_acf_sc_up)
        if aim_acf_sc_up >= tmp_acf_up:
            aim_acf_sc_up = tmp_acf_up
        aim_acf_sc_low = acf_mean - alpha * acf_std
        tmp_acf_low = acf_pdf.acf.quantile(0.025)
        if aim_acf_sc_low <= tmp_acf_low:
            aim_acf_sc_low = tmp_acf_low

        out_merge_df = out_merge_df.query("acf>=%f | acf <= %f" % (aim_acf_sc_up, aim_acf_sc_low)).reset_index(
            drop=True)

    if out_merge_df.shape[0] < 1:
        print("except error")
        out_merge_df = merge_fft_acf_res(aim_fft_f_pdf, acf_pdf).reset_index(drop=True)

    merge_index = list(out_merge_df.index)

    using_step = x_pdf.shape[0]

    input_ts = [((using_step + tstep) / fs) for tstep in range(win_predict)]

    f0 = f_list[0]
    p0 = p_list[0]
    total_results = [p0 for j in range(win_predict)]

    for i in range(len(merge_index)):
        #         print(out_merge_df['freq'][merge_index[i]])
        #         input_phase = out_merge_df['phase'][merge_index[i]] + (using_step*2*pi*out_merge_df['freq'][merge_index[i]] / 2*pi - int(using_step* / 2*pi))*2pi
        tmp_result = compute_sin_signal(A=out_merge_df['power'][merge_index[i]],
                                        freq=out_merge_df['freq'][merge_index[i]],
                                        phase=out_merge_df['phase'][merge_index[i]], ts=input_ts)

        #         print(tmp_result)
        for j in range(len(tmp_result)):
            total_results[j] += tmp_result[j]

    #         print(total_results)

    return fft_f_pdf, acf_pdf, out_merge_df, total_results, f0, p0


# %%


def compute_dtw(query_data,template_data,dist_method='euclidean',
    step_pattern='symmetric2'):
    dtw_distance_result = dtaidistance.dtw.distance_fast(query_data,template_data)
    return dtw_distance_result
#         alignment_result = dtw.dtw(query, template, keep_internals=True)
#     alignment_result = dtw.dtw(query_data,template_data,keep_internals=True,dist_method=dist_method,
#                               step_pattern=step_pattern)
#     return alignment_result.distance
#     alignment.distance
#     dtw.dtw()

# 通过与真实数据的MSE来进行判断：
def predict_multiple_results(y_true, forecast, mean, std, tail=-1):
    #     'yhat_lower', 'yhat_upper'
    if tail <= 0:
        y_predict = forecast['yhat']
        y_upper_predict = forecast['yhat_upper']
        y_lower_predict = forecast['yhat_lower']
    else:
        y_predict = forecast['yhat'][forecast.shape[0] - tail:]
        y_upper_predict = forecast['yhat_upper'][forecast.shape[0] - tail:]
        y_lower_predict = forecast['yhat_lower'][forecast.shape[0] - tail:]
    mseloss = MSE(y_true=y_true['y'] * std + mean, y_pred=y_predict * std + mean)
    lower_mseloss = MSE(y_true=y_true['y'] * std + mean, y_pred=y_lower_predict * std + mean)
    upper_mseloss = MSE(y_true=y_true['y'] * std + mean, y_pred=y_upper_predict * std + mean)
    return mseloss, lower_mseloss, upper_mseloss


# %%

def predict_single_results(y_true, forecast, mean, std, tail=-1):
    #     'yhat_lower', 'yhat_upper'
    if tail <= 0:
        y_predict = forecast
    else:
        y_predict = forecast[forecast.shape[0] - tail:]

    mseloss = MSE(y_true=y_true * std + mean, y_pred=y_predict * std + mean)

    return mseloss


# %%

def compute_relative_error(mseloss, mean_value):
    relative_res = math.sqrt(mseloss) / mean_value
    return relative_res


# %%

def initial_filter(step=3, aim_freq=1.2, sample_rate=1440, filter_type='lowpass'):
    aim_freq_rel = 2 * aim_freq / sample_rate
    b, a = signal.butter(step, aim_freq_rel, filter_type)
    return b, a


# %%

# 通过与真实数据的MSE来进行判断：
def predict_multiple_results_abs(y_true, forecast, mean, std, tail=-1):
    #     'yhat_lower', 'yhat_upper'
    if tail <= 0:
        y_predict = forecast['yhat']
        y_upper_predict = forecast['yhat_upper']
        y_lower_predict = forecast['yhat_lower']
    else:
        y_predict = forecast['yhat'][forecast.shape[0] - tail:]
        y_upper_predict = forecast['yhat_upper'][forecast.shape[0] - tail:]
        y_lower_predict = forecast['yhat_lower'][forecast.shape[0] - tail:]
    mseloss = MSE(y_true=y_true['y'], y_pred=y_predict)
    lower_mseloss = MSE(y_true=y_true['y'], y_pred=y_lower_predict)
    upper_mseloss = MSE(y_true=y_true['y'], y_pred=y_upper_predict)
    return mseloss, lower_mseloss, upper_mseloss


# %%

# 通过与真实数据的MSE来进行判断：
def predict_single_results_abs(y_true, forecast, mean, std, tail=-1, column='yhat1440'):
    #     'yhat_lower', 'yhat_upper'
    if tail <= 0:
        y_predict = forecast[column]
    #         y_upper_predict = forecast['yhat_upper']
    #         y_lower_predict = forecast['yhat_lower']
    else:
        y_predict = forecast[column][forecast.shape[0] - tail:]
    #         y_upper_predict = forecast['yhat_upper'][forecast.shape[0]-tail:]
    #         y_lower_predict = forecast['yhat_lower'][forecast.shape[0]-tail:]
    mseloss = MSE(y_true=y_true['y'], y_pred=y_predict)
    #     lower_mseloss = MSE(y_true=y_true['y'],y_pred=y_lower_predict)
    #     upper_mseloss = MSE(y_true=y_true['y'],y_pred=y_upper_predict)
    # ,lower_mseloss,upper_mseloss
    return mseloss


# %%

def raw_predict_qps_results(predict_model, future_step, freq='min', except_train=True):
    future = predict_model.make_future_dataframe(periods=future_step, freq=freq)
    future['cap'] = 3
    print(future.shape)
    # future = prophet_model.make_future_dataframe(periods=1440, freq='min')
    # print(future.shape)
    # future['cap'] =3
    if except_train:
        forecast = predict_model.predict(future[future.columns.to_list()][future.shape[0] - future_step:])
    else:
        forecast = predict_model.predict(future[future.columns.to_list()])
    return forecast




def initial_Prophet(holidays=pd.DataFrame(), holidays_prior_scale=9.0, seasonality_prior_scale=9.5,
                    growth='linear', n_changepoints=25, changepoint_range=0.85, changepoint_prior_scale=0.05,
                    weekly_seasonality=False, week_prior_scale=0.35, uncertainty_samples=1000):
    if holidays.shape[0] > 0:
        print("holidays")
        if seasonality_prior_scale > 0:
            if weekly_seasonality:
                prophet_model = Prophet(holidays=holidays, holidays_prior_scale=holidays_prior_scale,
                                        seasonality_prior_scale=seasonality_prior_scale, growth=growth,
                                        n_changepoints=n_changepoints,
                                        changepoint_range=changepoint_range,
                                        changepoint_prior_scale=changepoint_prior_scale,
                                        weekly_seasonality=weekly_seasonality, uncertainty_samples=uncertainty_samples)
                prophet_model.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=week_prior_scale)

            else:
                prophet_model = Prophet(holidays=holidays, holidays_prior_scale=holidays_prior_scale,
                                        seasonality_prior_scale=seasonality_prior_scale, growth=growth,
                                        n_changepoints=n_changepoints,
                                        changepoint_range=changepoint_range,
                                        changepoint_prior_scale=changepoint_prior_scale,
                                        uncertainty_samples=uncertainty_samples)
        else:
            if weekly_seasonality:
                prophet_model = Prophet(holidays=holidays, holidays_prior_scale=holidays_prior_scale,
                                        growth=growth, n_changepoints=n_changepoints,
                                        changepoint_range=changepoint_range,
                                        changepoint_prior_scale=changepoint_prior_scale,
                                        weekly_seasonality=weekly_seasonality, uncertainty_samples=uncertainty_samples)
                prophet_model.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=week_prior_scale)
            else:
                prophet_model = Prophet(holidays=holidays, holidays_prior_scale=holidays_prior_scale, growth=growth,
                                        n_changepoints=n_changepoints,
                                        changepoint_range=changepoint_range,
                                        changepoint_prior_scale=changepoint_prior_scale,
                                        uncertainty_samples=uncertainty_samples)
    else:
        if seasonality_prior_scale > 0:
            if weekly_seasonality:
                prophet_model = Prophet(seasonality_prior_scale=seasonality_prior_scale, growth=growth,
                                        n_changepoints=n_changepoints,
                                        changepoint_range=changepoint_range,
                                        changepoint_prior_scale=changepoint_prior_scale,
                                        weekly_seasonality=weekly_seasonality, uncertainty_samples=uncertainty_samples)
                prophet_model.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=week_prior_scale)
            else:
                prophet_model = Prophet(growth=growth, n_changepoints=n_changepoints,
                                        changepoint_range=changepoint_range,
                                        changepoint_prior_scale=changepoint_prior_scale,
                                        uncertainty_samples=uncertainty_samples)
        else:
            if weekly_seasonality:
                prophet_model = Prophet(holidays=holidays, holidays_prior_scale=holidays_prior_scale,
                                        seasonality_prior_scale=seasonality_prior_scale, growth=growth,
                                        n_changepoints=n_changepoints,
                                        changepoint_range=changepoint_range,
                                        changepoint_prior_scale=changepoint_prior_scale,
                                        weekly_seasonality=weekly_seasonality, uncertainty_samples=uncertainty_samples)
                prophet_model.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=week_prior_scale)

            else:
                prophet_model = Prophet(holidays=holidays, holidays_prior_scale=holidays_prior_scale,
                                        seasonality_prior_scale=seasonality_prior_scale, growth=growth,
                                        n_changepoints=n_changepoints,
                                        changepoint_range=changepoint_range,
                                        changepoint_prior_scale=changepoint_prior_scale,
                                        uncertainty_samples=uncertainty_samples)

    return prophet_model
#      seasonality_prior_scale=10.0,
#     holidays_prior_scale=10.0,
#     changepoint_prior_scale=0.05,




def find_lowest_dtw(x_data, win=30, history_len=1440, predict_len=60, sample_rate=1440, event_num=7, daliy_event_num=2,
                    dtw_length=1440, inter_step=5):
    tmp_days = math.floor((x_data.shape[0] - predict_len) / sample_rate)
    input_data = x_data.copy().reset_index(drop=True)
    res_data = input_data.tail(history_len + predict_len).reset_index(drop=True)

    res_labels = []
    res_trains = []

    #     res_dates = []

    stat_trains = []
    stat_labels = []

    #     stat_dates = []

    res_labels.append(res_data.tail(predict_len)[['ds', 'res_smooth']].to_numpy())
    res_trains.append(res_data.head(history_len)[['ds', 'res_smooth']].to_numpy())
    template_data = res_data.head(history_len)['res_smooth'].to_numpy()
    print("template data raw shape:")
    print(template_data.shape)
    if dtw_length > 0 and dtw_length < history_len:
        template_data = template_data[(history_len - dtw_length):]
        print("template data real shape:")
        print(template_data.shape)
    #     res_dates.append()

    print("res trains shape:")
    print(res_trains[0].shape)
    print("res labels shape:")
    print(res_labels[0].shape)

    stat_labels.append(res_data.tail(predict_len)[['ds', 'res_stat']].to_numpy())
    stat_trains.append(res_data.head(history_len)[['ds', 'res_stat']].to_numpy())

    global_start_time = x_data.ds.min()

    base_end_time = res_data.head(history_len).ds.max()

    print(base_end_time)

    #     # pd.to_datetime(int(c1.to_numpy())/1e9+1440*60 + 30*60,unit='s')
    poitential_data = {}
    for j in range(1, tmp_days):
        tmp_poitential = {}
        tmp_position = {}
        #         time_range = [int(c1.to_numpy())/1e9-j*sample_rate*60 + k*60 for k in range(-win,win+1)]

        for w in range(-win, win + 1, inter_step):
            t = int(base_end_time.to_numpy()) / 1e9 - j * sample_rate * 60 + w * 60
            #             print(t)
            #             print(w)
            date_t = pd.to_datetime(t, unit='s')
            #             if w == win:
            #             print(date_t)
            aim_data = input_data[input_data.ds <= date_t].sort_values(['ds']).reset_index(drop=True)
            #             print(aim_data.shape)
            if aim_data.shape[0] < history_len:
                continue

            input_pre_data = aim_data.tail(history_len)[['ds', 'res_smooth']].to_numpy()
            if dtw_length > 0 and dtw_length < history_len:
                input_pre_query = aim_data.tail(dtw_length)['res_smooth'].to_numpy()
            else:
                input_pre_query = aim_data.tail(history_len)['res_smooth'].to_numpy()
            #             print("input_pre_query shape:")
            #             print(input_pre_query.shape)
            #             print("input pre data shape:")
            #             print(input_pre_data.shape)
            #             print(template_data)
            dtw_distance = compute_dtw(query_data=input_pre_query, template_data=template_data)

            #             print(dtw_distance)
            #             print(tmp_poitential.keys())
            if dtw_distance in tmp_poitential.keys():
                if np.abs(w) < tmp_position[dtw_distance]:
                    input_pre_ds_end = aim_data.tail(history_len).ds.max()
                    input_pre_label = \
                    input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                        predict_len)[['ds', "res_smooth"]].to_numpy()
                    input_pre_stat_data = aim_data.tail(history_len)[['ds', 'res_stat']].to_numpy()
                    input_pre_stat_label = \
                    input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                        predict_len)[['ds', "res_stat"]].to_numpy()
                    tmp_position[dtw_distance] = np.abs(w)
                    tmp_poitential[dtw_distance]['res_train'] = input_pre_data
                    tmp_poitential[dtw_distance]['res_label'] = input_pre_label
                    tmp_poitential[dtw_distance]['stat_train'] = input_pre_stat_data
                    tmp_poitential[dtw_distance]['stat_label'] = input_pre_stat_label
                else:
                    continue
            else:
                if len(tmp_poitential.keys()) < daliy_event_num:
                    input_pre_ds_end = aim_data.tail(history_len).ds.max()
                    input_pre_label = \
                    input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                        predict_len)[['ds', 'res_smooth']].to_numpy()
                    input_pre_stat_data = aim_data.tail(history_len)[['ds', 'res_stat']].to_numpy()
                    input_pre_stat_label = \
                    input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                        predict_len)[['ds', 'res_stat']].to_numpy()
                    tmp_position[dtw_distance] = np.abs(w)

                    tmp_poitential[dtw_distance] = {}
                    tmp_poitential[dtw_distance]['res_train'] = input_pre_data
                    tmp_poitential[dtw_distance]['res_label'] = input_pre_label
                    tmp_poitential[dtw_distance]['stat_train'] = input_pre_stat_data
                    tmp_poitential[dtw_distance]['stat_label'] = input_pre_stat_label
                else:
                    tmp_key_los = np.max(list(tmp_poitential.keys()))
                    if dtw_distance < tmp_key_los:
                        input_pre_ds_end = aim_data.tail(history_len).ds.max()
                        input_pre_label = \
                        input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                            predict_len)[['ds', 'res_smooth']].to_numpy()
                        input_pre_stat_data = aim_data.tail(history_len)[['ds', 'res_stat']].to_numpy()
                        input_pre_stat_label = \
                        input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                            predict_len)[['ds', 'res_stat']].to_numpy()
                        tmp_position[dtw_distance] = np.abs(w)

                        tmp_poitential[dtw_distance] = {}
                        tmp_poitential[dtw_distance]['res_train'] = input_pre_data
                        tmp_poitential[dtw_distance]['res_label'] = input_pre_label
                        tmp_poitential[dtw_distance]['stat_train'] = input_pre_stat_data
                        tmp_poitential[dtw_distance]['stat_label'] = input_pre_stat_label

                        tmp_poitential.pop(tmp_key_los)
                        tmp_position.pop(tmp_key_los)

        for k in tmp_poitential.keys():
            #             print(k)
            #             poitential_data.append({ 'res_train':tmp_poitential[k]['res_train'],
            #                                      'res_label':tmp_poitential[k]['res_label'],
            #                                      'stat_train': tmp_poitential[k]['stat_train'],
            #                                      'stat_label':tmp_poitential[k]['stat_label'],
            #                                        'date':j,
            #                                        'dtw':k})
            if j not in poitential_data.keys():
                poitential_data[j] = {}

            poitential_data[j][k] = {}
            poitential_data[j][k]['res_train'] = tmp_poitential[k]['res_train']
            poitential_data[j][k]['res_label'] = tmp_poitential[k]['res_label']
            poitential_data[j][k]['stat_train'] = tmp_poitential[k]['stat_train']
            poitential_data[j][k]['stat_label'] = tmp_poitential[k]['stat_label']
    #             poitential_data[k]['date'] = j

    aim_dates = list(poitential_data.keys())
    #     aim_dates.sort()

    find_index = []
    total_poitent = []

    for ad in aim_dates:
        for po_dtw in poitential_data[ad].keys():
            total_poitent.append((ad, po_dtw))

    find_temp = 0
    while ((len(find_index) < event_num) and (len(total_poitent) > 0)):
        ad_dtws = {}
        for item in total_poitent:
            if item[0] not in ad_dtws.keys():
                ad_dtws[item[0]] = []
            ad_dtws[item[0]].append(item[1])

        find_in_this_time = {}
        for ad in ad_dtws.keys():
            ad_dtws[ad].sort()
            if ad_dtws[ad][0] in find_in_this_time.keys():
                if ad < find_in_this_time[ad_dtws[ad][0]]:
                    find_in_this_time[ad_dtws[ad][0]] = ad
            else:
                find_in_this_time[ad_dtws[ad][0]] = ad
        po_dtws = list(find_in_this_time.keys())
        po_dtws.sort()
        for pdt in po_dtws:
            if len(find_index) >= event_num:
                break
            if len(total_poitent) <= 0:
                break
            find_index.append((find_in_this_time[pdt], pdt))
            total_poitent.remove((find_in_this_time[pdt], pdt))

        if len(find_index) >= event_num:
            break

        if len(total_poitent) <= 0:
            break

    #     print(find_index)
    final_res = {}

    for fi in find_index:
        po_data = poitential_data[fi[0]][fi[1]]
        po_time = po_data['res_train'][history_len - 1][0]
        #         print(po_time)
        final_res[po_time] = poitential_data[fi[0]][fi[1]]

    final_keys = list(final_res.keys())
    final_keys.sort()

    #     print(final_keys)

    for i in range(len(final_keys) - 1, -1, -1):
        #         print(final_keys[i])
        res_trains.append(final_res[final_keys[i]]['res_train'])
        res_labels.append(final_res[final_keys[i]]['res_label'])

        stat_trains.append(final_res[final_keys[i]]['stat_train'])
        stat_labels.append(final_res[final_keys[i]]['stat_label'])

    print(res_trains[-1].shape)
    print(res_labels[-1].shape)
    print(len(res_trains))

    return res_trains, res_labels, stat_trains, stat_labels


def find_lowest_data_base(x_data, win=30, history_len=1440, predict_len=60, sample_rate=1440, event_num=7,
                          daliy_event_num=2, inter_step=5):
    tmp_days = math.floor((x_data.shape[0] - predict_len) / sample_rate)
    input_data = x_data.copy().reset_index(drop=True)
    res_data = input_data.tail(history_len + predict_len).reset_index(drop=True)

    res_labels = []
    res_trains = []

    prop_tmp_labels = []
    prop_tmp_trains = []

    y_tmp_labels = []
    y_tmp_trains = []

    #     res_dates = []

    stat_trains = []
    stat_labels = []

    #     stat_dates = []

    res_labels.append(res_data.tail(predict_len)[['ds', 'res_smooth']].to_numpy())
    res_trains.append(res_data.head(history_len)[['ds', 'res_smooth']].to_numpy())
    template_data = res_data.head(history_len)['res_smooth'].to_numpy()
    #     res_dates.append()

    stat_labels.append(res_data.tail(predict_len)[['ds', 'res_stat']].to_numpy())
    stat_trains.append(res_data.head(history_len)[['ds', 'res_stat']].to_numpy())

    y_tmp_labels.append(res_data.tail(predict_len)[['ds', 'y']].to_numpy())
    y_tmp_trains.append(res_data.head(history_len)[['ds', 'y']].to_numpy())

    prop_tmp_labels.append(res_data.tail(predict_len)[['ds', 'yhat']].to_numpy())
    prop_tmp_trains.append(res_data.head(history_len)[['ds', 'yhat']].to_numpy())

    global_start_time = x_data.ds.min()

    base_end_time = res_data.head(history_len).ds.max()

    print(base_end_time)

    print(res_trains[0].shape)
    print(res_labels[0].shape)
    print(len(res_trains))

    return res_trains, res_labels, stat_trains, stat_labels, prop_tmp_trains, prop_tmp_labels, y_tmp_trains, y_tmp_labels



def build_resdiual(history_data, test_data, prop_model, win_prop=1440, min_cycle=14, start_id=1440 * 7, end_id=-1,
                   up_step=1, sample_rate=1440, lowpass_upper=24, filter_step=3,
                   predict_len=60, dtw_win=30, train_history_len=1440, event_num=7, daliy_event_num=2, dtw_length=480):
    prop_res = raw_predict_qps_results(prop_model, future_step=win_prop, freq='min', except_train=False)

    time01 = time.time()
    raw_result = prop_res.copy().sort_values(['ds']).reset_index(drop=True)
    raw_result = raw_result[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]

    print(raw_result.shape)
    print(raw_result.dropna().shape)

    raw_predict = raw_result.copy().tail(win_prop).reset_index(drop=True)

    input_history = history_data.copy().sort_values(['ds']).reset_index(drop=True)

    res_train = raw_result.head(prop_res.shape[0] - win_prop).reset_index(drop=True)
    res_train['res_total'] = input_history['y'] - res_train['yhat']
    # - res_train['yhat']
    res_train['y'] = input_history.copy().reset_index(drop=True)['y']

    print(res_train.shape)
    print(res_train.dropna().shape)

    res_test = raw_predict.copy().sort_values(['ds']).reset_index(drop=True)

    input_test = test_data.copy().sort_values(['ds']).reset_index(drop=True)
    res_test['res_total'] = input_test['y'] - res_test['yhat']
    # - res_test['yhat']
    res_test['y'] = input_test.copy().reset_index(drop=True)['y']

    res_train = res_train.sort_values(['ds']).reset_index(drop=True)
    res_test = res_test.sort_values(['ds']).reset_index(drop=True)

    print(res_test.shape)
    print(res_test.dropna().shape)

    #     res_train = res_train.tail(res_train.shape[0]-start_id).sort_values(['ds']).reset_index(drop=True)

    train_results = {'history': [], 'predict': []}
    y_results = {'history': [], 'predict': []}
    prop_results = {'history': [], 'predict': []}
    train_stat = {'history': [], 'predict': []}

    end_step = np.min([end_id, res_train.shape[0] - predict_len])

    if end_id < 0:
        end_step = res_train.shape[0] - predict_len

    for step in range(start_id, end_step, up_step):
        time11 = time.time()
        filter_length = np.min([step, min_cycle * sample_rate])
        if filter_length + dtw_win < step:
            filter_length = filter_length + dtw_win
        tmp_to_filter_data = res_train[step - filter_length:step + predict_len].copy()
        tmp_to_filter_data = tmp_to_filter_data.reset_index(drop=True)

        bk, ak = initial_filter(step=filter_step, aim_freq=lowpass_upper, sample_rate=sample_rate,
                                filter_type='lowpass')

        filteddata = signal.filtfilt(bk, ak, tmp_to_filter_data.res_total.to_numpy())
        print(tmp_to_filter_data.res_total.shape)
        print(tmp_to_filter_data.res_total.dropna().shape)
        print(filteddata)
        print(filter_length)
        tmp_to_filter_data['res_smooth'] = pd.Series(list(filteddata))
        tmp_to_filter_data['res_stat'] = tmp_to_filter_data['res_total'] - tmp_to_filter_data['res_smooth']
        #         (x_data,win=30,history_len=1440,predict_len=60,sample_rate=1440,event_num=7,daliy_event_num=2,inter_step=5)
        # train_his, train_labels, stat_his, stat_labels = find_lowest_dtw(x_data=tmp_to_filter_data, win=dtw_win,
        #                                                                  history_len=train_history_len,
        #                                                                  predict_len=predict_len,
        #                                                                  sample_rate=sample_rate,
        #                                                                  event_num=event_num,
        #                                                                  daliy_event_num=daliy_event_num,
        #                                                                  dtw_length=dtw_length)

        train_his, train_labels, y_his, y_labels, prop_his, prop_labels,stat_his, stat_labels = find_lowest_dtw2(x_data=tmp_to_filter_data,
                                                                                           win=dtw_win,
                                                                                           history_len=train_history_len,
                                                                                           predict_len=predict_len,
                                                                                           sample_rate=sample_rate,
                                                                                           event_num=event_num,
                                                                                           daliy_event_num=daliy_event_num,
                                                                                           dtw_length=dtw_length)

        train_results['history'].append(np.array(train_his))
        train_results['predict'].append(np.array(train_labels))
        train_stat['history'].append(np.array(stat_his))
        train_stat['predict'].append(np.array(stat_labels))

        y_results['history'].append(np.array(y_his))
        y_results['predict'].append(np.array(y_labels))

        prop_results['history'].append(np.array(prop_his))
        prop_results['predict'].append(np.array(prop_labels))



        print(time.time() - time11)

    test_results = {'history': [], 'predict': []}
    test_stat = {'history': [], 'predict': []}
    y_test_results = {'history': [], 'predict': []}
    prop_test_results = {'history': [], 'predict': []}

    res_total_data = pd.concat([res_train.copy().reset_index(drop=True), res_test.copy().reset_index(drop=True)],
                               axis=0).reset_index(drop=True)
    res_total_data = res_total_data.sort_values(['ds']).reset_index(drop=True)

    if end_id > 0:
        train_results['history'] = np.array(train_results['history'])
        train_results['predict'] = np.array(train_results['predict'])
        train_stat['history'] = np.array(train_stat['history'])
        train_stat['predict'] = np.array(train_stat['predict'])

        y_results['history'] = np.array(y_results['history'])
        y_results['predict'] = np.array(y_results['predict'])
        prop_results['history'] = np.array(prop_results['history'])
        prop_results['predict'] = np.array(prop_results['predict'])
        # train_results, y_results, prop_results
        return train_results, y_results, prop_results, train_stat

    for step in range(res_train.shape[0], res_total_data.shape[0] - predict_len, up_step):
        filter_length = np.min([step, min_cycle * sample_rate])
        if filter_length + dtw_win < step:
            filter_length = filter_length + dtw_win

        tmp_to_filter_data = res_total_data[step - filter_length:step + predict_len].copy()
        tmp_to_filter_data = tmp_to_filter_data.reset_index(drop=True)

        bk, ak = initial_filter(step=filter_step, aim_freq=lowpass_upper, sample_rate=sample_rate,
                                filter_type='lowpass')

        filteddata = signal.filtfilt(bk, ak, tmp_to_filter_data.res_total.to_numpy())
        print(filter_length)
        tmp_to_filter_data['res_smooth'] = pd.Series(list(filteddata))
        tmp_to_filter_data['res_stat'] = tmp_to_filter_data['res_total'] - tmp_to_filter_data['res_smooth']

        #         (x_data,win=30,history_len=1440,predict_len=60,sample_rate=1440,event_num=7,daliy_event_num=2,inter_step=5)

        train_his, train_labels, y_his, y_labels, prop_his, prop_labels,stat_his, stat_labels = find_lowest_dtw2(x_data=tmp_to_filter_data,
                                                                                           win=dtw_win,
                                                                                           history_len=train_history_len,
                                                                                           predict_len=predict_len,
                                                                                           sample_rate=sample_rate,
                                                                                           event_num=event_num,
                                                                                           daliy_event_num=daliy_event_num,
                                                                                           dtw_length=dtw_length)

        test_results['history'].append(np.array(train_his))
        test_results['predict'].append(np.array(train_labels))
        test_stat['history'].append(np.array(stat_his))
        test_stat['predict'].append(np.array(stat_labels))

        y_test_results['history'].append(np.array(y_his))
        y_test_results['predict'].append(np.array(y_labels))

        prop_test_results['history'].append(np.array(prop_his))
        prop_test_results['predict'].append(np.array(prop_labels))



    train_results['history'] = np.array(train_results['history'])
    train_results['predict'] = np.array(train_results['predict'])

    train_stat['history'] = np.array(train_stat['history'])
    train_stat['predict'] = np.array(train_stat['predict'])

    y_results['history'] = np.array(y_results['history'])
    y_results['predict'] = np.array(y_results['predict'])

    prop_results['history'] = np.array(prop_results['history'])
    prop_results['predict'] = np.array(prop_results['predict'])

    test_results['history'] = np.array(test_results['history'])
    test_results['predict'] = np.array(test_results['predict'])

    test_stat['history'] = np.array(test_stat['history'])
    test_stat['predict'] = np.array(test_stat['predict'])

    y_test_results['history'] = np.array(y_test_results['history'])
    y_test_results['predict'] = np.array(y_test_results['predict'])

    prop_test_results['history'] = np.array(prop_test_results['history'])
    prop_test_results['predict'] = np.array(prop_test_results['predict'])



    return train_results, y_results, prop_results,train_stat, test_results,y_test_results, prop_test_results ,test_stat

def find_lowest_dtw2(x_data, win=10, history_len=1440, predict_len=60, sample_rate=1440, event_num=7, daliy_event_num=2,
                     dtw_length=480, inter_step=5, seconds=60):
    tmp_days = math.floor((x_data.shape[0] - predict_len) / sample_rate)
    input_data = x_data.copy().reset_index(drop=True)
    res_data = input_data.tail(history_len + predict_len).reset_index(drop=True)
    print("find days is: %d" % tmp_days)

    res_labels = []
    res_trains = []

    y0_trains = []
    y0_labels = []

    prop0_trains = []
    prop0_labels = []

    #     res_dates = []

    stat_trains = []
    stat_labels = []

    #     stat_dates = []

    res_labels.append(res_data.tail(predict_len)[['ds', 'res_smooth']].to_numpy())
    res_trains.append(res_data.head(history_len)[['ds', 'res_smooth']].to_numpy())
    template_data = res_data.head(history_len)['res_smooth'].to_numpy()
    print("template data raw shape:")
    print(template_data.shape)
    if dtw_length > 0 and dtw_length < history_len:
        template_data = template_data[(history_len - dtw_length):]
        print("template data real shape:")
        print(template_data.shape)
    #     res_dates.append()

    print("res trains shape:")
    print(res_trains[0].shape)
    print("res labels shape:")
    print(res_labels[0].shape)

    stat_labels.append(res_data.tail(predict_len)[['ds', 'res_stat']].to_numpy())
    stat_trains.append(res_data.head(history_len)[['ds', 'res_stat']].to_numpy())

    y0_labels.append(res_data.tail(predict_len)[['ds', 'y']].to_numpy())
    y0_trains.append(res_data.head(history_len)[['ds', 'y']].to_numpy())

    prop0_labels.append(res_data.tail(predict_len)[['ds', 'yhat']].to_numpy())
    prop0_trains.append(res_data.head(history_len)[['ds', 'yhat']].to_numpy())

    global_start_time = x_data.ds.min()

    base_end_time = res_data.head(history_len).ds.max()

    print(base_end_time)

    #     # pd.to_datetime(int(c1.to_numpy())/1e9+1440*60 + 30*60,unit='s')
    poitential_data = {}
    for j in range(1, tmp_days):
        tmp_poitential = {}
        tmp_position = {}
        #         time_range = [int(c1.to_numpy())/1e9-j*sample_rate*60 + k*60 for k in range(-win,win+1)]

        for w in range(-win, win + 1, inter_step):
            t = int(base_end_time.to_numpy()) / 1e9 - j * sample_rate * seconds + w * 60
            #             print(t)
            #             print(w)
            date_t = pd.to_datetime(t, unit='s')
            #             if w == win:
            #             print(date_t)
            aim_data = input_data[input_data.ds <= date_t].sort_values(['ds']).reset_index(drop=True)
            #             print(aim_data.shape)
            if aim_data.shape[0] < history_len:
                continue

            input_pre_data = aim_data.tail(history_len)[['ds', 'res_smooth']].to_numpy()
            if dtw_length > 0 and dtw_length < history_len:
                input_pre_query = aim_data.tail(dtw_length)['res_smooth'].to_numpy()
            else:
                input_pre_query = aim_data.tail(history_len)['res_smooth'].to_numpy()
            #             print("input_pre_query shape:")
            #             print(input_pre_query.shape)
            #             print("input pre data shape:")
            #             print(input_pre_data.shape)
            #             print(template_data)
            dtw_distance = compute_dtw(query_data=input_pre_query, template_data=template_data)

            #             print(dtw_distance)
            #             print(tmp_poitential.keys())
            if dtw_distance in tmp_poitential.keys():
                if np.abs(w) < tmp_position[dtw_distance]:
                    input_pre_ds_end = aim_data.tail(history_len).ds.max()
                    input_pre_label = \
                        input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                            predict_len)[['ds', "res_smooth"]].to_numpy()

                    input_pre_prop_data = aim_data.tail(history_len)[['ds', 'yhat']].to_numpy()
                    input_pre_prop_label = \
                    input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                        predict_len)[['ds', 'yhat']].to_numpy()

                    input_pre_y_data = aim_data.tail(history_len)[['ds', 'y']].to_numpy()
                    input_pre_y_label = \
                    input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                        predict_len)[['ds', 'y']].to_numpy()

                    input_pre_stat_data = aim_data.tail(history_len)[['ds', 'res_stat']].to_numpy()
                    input_pre_stat_label = \
                    input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                        predict_len)[['ds', "res_stat"]].to_numpy()
                    tmp_position[dtw_distance] = np.abs(w)
                    tmp_poitential[dtw_distance]['res_train'] = input_pre_data
                    tmp_poitential[dtw_distance]['res_label'] = input_pre_label

                    tmp_poitential[dtw_distance]['prop_train'] = input_pre_prop_data
                    tmp_poitential[dtw_distance]['prop_label'] = input_pre_prop_label

                    tmp_poitential[dtw_distance]['y_train'] = input_pre_y_data
                    tmp_poitential[dtw_distance]['y_label'] = input_pre_y_label

                    tmp_poitential[dtw_distance]['stat_train'] = input_pre_stat_data
                    tmp_poitential[dtw_distance]['stat_label'] = input_pre_stat_label
                else:
                    continue
            else:
                if len(tmp_poitential.keys()) < daliy_event_num:
                    input_pre_ds_end = aim_data.tail(history_len).ds.max()
                    input_pre_label = \
                        input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                            predict_len)[['ds', 'res_smooth']].to_numpy()
                    input_pre_stat_data = aim_data.tail(history_len)[['ds', 'res_stat']].to_numpy()
                    input_pre_stat_label = \
                    input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                        predict_len)[['ds', 'res_stat']].to_numpy()

                    input_pre_prop_data = aim_data.tail(history_len)[['ds', 'yhat']].to_numpy()
                    input_pre_prop_label = \
                    input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                        predict_len)[['ds', 'yhat']].to_numpy()

                    input_pre_y_data = aim_data.tail(history_len)[['ds', 'y']].to_numpy()
                    input_pre_y_label = \
                    input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                        predict_len)[['ds', 'y']].to_numpy()

                    tmp_position[dtw_distance] = np.abs(w)

                    tmp_poitential[dtw_distance] = {}
                    tmp_poitential[dtw_distance]['res_train'] = input_pre_data
                    tmp_poitential[dtw_distance]['res_label'] = input_pre_label
                    tmp_poitential[dtw_distance]['stat_train'] = input_pre_stat_data
                    tmp_poitential[dtw_distance]['stat_label'] = input_pre_stat_label
                    tmp_poitential[dtw_distance]['prop_train'] = input_pre_prop_data
                    tmp_poitential[dtw_distance]['prop_label'] = input_pre_prop_label

                    tmp_poitential[dtw_distance]['y_train'] = input_pre_y_data
                    tmp_poitential[dtw_distance]['y_label'] = input_pre_y_label

                else:
                    tmp_key_los = np.max(list(tmp_poitential.keys()))
                    if dtw_distance < tmp_key_los:
                        input_pre_ds_end = aim_data.tail(history_len).ds.max()
                        input_pre_label = \
                            input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(
                                drop=True).head(
                                predict_len)[['ds', 'res_smooth']].to_numpy()
                        input_pre_prop_data = aim_data.tail(history_len)[['ds', 'yhat']].to_numpy()
                        input_pre_prop_label = \
                        input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                            predict_len)[['ds', 'yhat']].to_numpy()
                        input_pre_y_data = aim_data.tail(history_len)[['ds', 'y']].to_numpy()
                        input_pre_y_label = \
                        input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                            predict_len)[['ds', 'y']].to_numpy()

                        input_pre_stat_data = aim_data.tail(history_len)[['ds', 'res_stat']].to_numpy()
                        input_pre_stat_label = \
                        input_data[input_data.ds > input_pre_ds_end].sort_values(['ds']).reset_index(drop=True).head(
                            predict_len)[['ds', 'res_stat']].to_numpy()
                        tmp_position[dtw_distance] = np.abs(w)

                        tmp_poitential[dtw_distance] = {}

                        tmp_poitential[dtw_distance]['res_train'] = input_pre_data
                        tmp_poitential[dtw_distance]['res_label'] = input_pre_label

                        tmp_poitential[dtw_distance]['prop_train'] = input_pre_prop_data
                        tmp_poitential[dtw_distance]['prop_label'] = input_pre_prop_label

                        tmp_poitential[dtw_distance]['y_train'] = input_pre_y_data
                        tmp_poitential[dtw_distance]['y_label'] = input_pre_y_label
                        tmp_poitential[dtw_distance]['stat_train'] = input_pre_stat_data
                        tmp_poitential[dtw_distance]['stat_label'] = input_pre_stat_label

                        tmp_poitential.pop(tmp_key_los)
                        tmp_position.pop(tmp_key_los)

        for k in tmp_poitential.keys():
            #             print(k)
            #             poitential_data.append({ 'res_train':tmp_poitential[k]['res_train'],
            #                                      'res_label':tmp_poitential[k]['res_label'],
            #                                      'stat_train': tmp_poitential[k]['stat_train'],
            #                                      'stat_label':tmp_poitential[k]['stat_label'],
            #                                        'date':j,
            #                                        'dtw':k})
            if j not in poitential_data.keys():
                poitential_data[j] = {}

            poitential_data[j][k] = {}
            poitential_data[j][k]['res_train'] = tmp_poitential[k]['res_train']
            poitential_data[j][k]['res_label'] = tmp_poitential[k]['res_label']
            poitential_data[j][k]['y_train'] = tmp_poitential[k]['y_train']
            poitential_data[j][k]['y_label'] = tmp_poitential[k]['y_label']
            poitential_data[j][k]['prop_train'] = tmp_poitential[k]['prop_train']
            poitential_data[j][k]['prop_label'] = tmp_poitential[k]['prop_label']
            poitential_data[j][k]['stat_train'] = tmp_poitential[k]['stat_train']
            poitential_data[j][k]['stat_label'] = tmp_poitential[k]['stat_label']
    #             poitential_data[k]['date'] = j

    aim_dates = list(poitential_data.keys())
    #     aim_dates.sort()

    find_index = []
    total_poitent = []

    for ad in aim_dates:
        for po_dtw in poitential_data[ad].keys():
            total_poitent.append((ad, po_dtw))

    find_temp = 0
    while ((len(find_index) < event_num) and (len(total_poitent) > 0)):
        ad_dtws = {}
        for item in total_poitent:
            if item[0] not in ad_dtws.keys():
                ad_dtws[item[0]] = []
            ad_dtws[item[0]].append(item[1])

        find_in_this_time = {}
        for ad in ad_dtws.keys():
            ad_dtws[ad].sort()
            if ad_dtws[ad][0] in find_in_this_time.keys():
                if ad < find_in_this_time[ad_dtws[ad][0]]:
                    find_in_this_time[ad_dtws[ad][0]] = ad
            else:
                find_in_this_time[ad_dtws[ad][0]] = ad
        po_dtws = list(find_in_this_time.keys())
        po_dtws.sort()
        for pdt in po_dtws:
            if len(find_index) >= event_num:
                break
            if len(total_poitent) <= 0:
                break
            find_index.append((find_in_this_time[pdt], pdt))
            total_poitent.remove((find_in_this_time[pdt], pdt))

        if len(find_index) >= event_num:
            break

        if len(total_poitent) <= 0:
            break

    #     print(find_index)
    final_res = {}

    for fi in find_index:
        po_data = poitential_data[fi[0]][fi[1]]
        po_time = po_data['res_train'][history_len - 1][0]
        #         print(po_time)
        final_res[po_time] = poitential_data[fi[0]][fi[1]]

    final_keys = list(final_res.keys())
    final_keys.sort()

    #     print(final_keys)

    for i in range(len(final_keys) - 1, -1, -1):
        #         print(final_keys[i])
        res_trains.append(final_res[final_keys[i]]['res_train'])
        res_labels.append(final_res[final_keys[i]]['res_label'])

        prop0_trains.append(final_res[final_keys[i]]['prop_train'])
        prop0_labels.append(final_res[final_keys[i]]['prop_label'])

        y0_trains.append(final_res[final_keys[i]]['y_train'])
        y0_labels.append(final_res[final_keys[i]]['y_label'])

        stat_trains.append(final_res[final_keys[i]]['stat_train'])
        stat_labels.append(final_res[final_keys[i]]['stat_label'])

    print(res_trains[-1].shape)
    print(res_labels[-1].shape)
    print(len(res_trains))

    #     # train_his, train_labels, y_his, y_labels,prop_his,prop_labels,stat_his, stat_labels

    return res_trains, res_labels, y0_trains, y0_labels, prop0_trains, prop0_labels,stat_trains,stat_labels



def build_resdiual2(history_data, test_data, prop_model, win_prop=96 * 7, min_cycle=14, start_id=96 * 14, end_id=-1,
                   up_step=1, sample_rate=96, lowpass_upper=48, filter_step=3,
                   predict_len=96, dtw_win=30, train_history_len=96 * 7, event_num=7, daliy_event_num=2,
                   dtw_length=96 * 4):
    prop_res = raw_predict_qps_results(prop_model, future_step=win_prop, freq='15min', except_train=False)

    time01 = time.time()
    raw_result = prop_res.copy().sort_values(['ds']).reset_index(drop=True)
    raw_result = raw_result[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
    print("raw_result shape:")
    print(raw_result.shape)
    print(raw_result.dropna().shape)
    print("expect raw results shape:")
    print(history_data.shape[0] + test_data.shape[0])

    raw_predict = raw_result.copy().tail(win_prop).reset_index(drop=True)

    input_history = history_data.copy().sort_values(['ds']).reset_index(drop=True)

    res_train = raw_result.head(prop_res.shape[0] - win_prop).reset_index(drop=True)
    res_train['res_total'] = input_history['y'] - res_train['yhat']
    # - res_train['yhat']
    res_train['y'] = input_history.copy().reset_index(drop=True)['y']

    print(res_train.shape)
    print(res_train.dropna().shape)

    res_test = raw_predict.copy().sort_values(['ds']).reset_index(drop=True)

    input_test = test_data.copy().sort_values(['ds']).reset_index(drop=True)
    res_test['res_total'] = input_test['y'] - res_test['yhat']
    # - res_test['yhat']
    res_test['y'] = input_test.copy().reset_index(drop=True)['y']

    res_train = res_train.sort_values(['ds']).reset_index(drop=True)
    res_test = res_test.sort_values(['ds']).reset_index(drop=True)

    print(res_test.shape)
    print(res_test.dropna().shape)

    #     res_train = res_train.tail(res_train.shape[0]-start_id).sort_values(['ds']).reset_index(drop=True)

    train_results = {'history': [], 'predict': []}
    y_results = {'history': [], 'predict': []}
    prop_results = {'history': [], 'predict': []}
    train_stat = {'history': [], 'predict': []}

    end_step = np.min([end_id, res_train.shape[0] - predict_len])

    if end_id < 0:
        end_step = res_train.shape[0] - predict_len

    for step in range(start_id, end_step, up_step):
        time11 = time.time()
        filter_length = np.min([step, min_cycle * sample_rate])
        if filter_length + dtw_win < step:
            filter_length = filter_length + dtw_win
        tmp_to_filter_data = res_train[step - filter_length:step + predict_len].copy()
        tmp_to_filter_data = tmp_to_filter_data.reset_index(drop=True)

        #         bk, ak = initial_filter(step=filter_step, aim_freq=lowpass_upper, sample_rate=sample_rate,
        #                                 filter_type='lowpass')

        #         filteddata = signal.filtfilt(bk, ak, tmp_to_filter_data.res_total.to_numpy())
        filteddata = tmp_to_filter_data.res_total.to_numpy()
        print(tmp_to_filter_data.res_total.shape)
        print(tmp_to_filter_data.res_total.dropna().shape)
        # print(filteddata)
        print(filter_length)
        tmp_to_filter_data['res_smooth'] = pd.Series(list(filteddata))
        tmp_to_filter_data['res_stat'] = tmp_to_filter_data['res_total'] - tmp_to_filter_data['res_smooth']
        #         (x_data,win=30,history_len=1440,predict_len=60,sample_rate=1440,event_num=7,daliy_event_num=2,inter_step=5)
        train_his, train_labels, y_his, y_labels, prop_his, prop_labels = find_lowest_dtw2(x_data=tmp_to_filter_data,
                                                                                           win=dtw_win,
                                                                                           history_len=train_history_len,
                                                                                           predict_len=predict_len,
                                                                                           sample_rate=sample_rate,
                                                                                           event_num=event_num,
                                                                                           daliy_event_num=daliy_event_num,
                                                                                           dtw_length=dtw_length)

        train_results['history'].append(np.array(train_his))
        train_results['predict'].append(np.array(train_labels))

        y_results['history'].append(np.array(y_his))
        y_results['predict'].append(np.array(y_labels))

        prop_results['history'].append(np.array(prop_his))
        prop_results['predict'].append(np.array(prop_labels))

        # train_stat['history'].append(np.array(stat_his))
        # train_stat['predict'].append(np.array(stat_labels))

        print(time.time() - time11)

    test_results = {'history': [], 'predict': []}
    test_stat = {'history': [], 'predict': []}
    y_test_results = {'history': [], 'predict': []}
    prop_test_results = {'history': [], 'predict': []}

    res_total_data = pd.concat([res_train.copy().reset_index(drop=True), res_test.copy().reset_index(drop=True)],
                               axis=0).reset_index(drop=True)
    res_total_data = res_total_data.sort_values(['ds']).reset_index(drop=True)

    if end_id > 0:
        train_results['history'] = np.array(train_results['history'])
        train_results['predict'] = np.array(train_results['predict'])
        y_results['history'] = np.array(y_results['history'])
        y_results['predict'] = np.array(y_results['predict'])
        prop_results['history'] = np.array(prop_results['history'])
        prop_results['predict'] = np.array(prop_results['predict'])
        return train_results, y_results, prop_results

    for step in range(res_train.shape[0], res_total_data.shape[0] - predict_len, up_step):
        filter_length = np.min([step, min_cycle * sample_rate])
        if filter_length + dtw_win < step:
            filter_length = filter_length + dtw_win

        tmp_to_filter_data = res_total_data[step - filter_length:step + predict_len].copy()
        tmp_to_filter_data = tmp_to_filter_data.reset_index(drop=True)

        # bk, ak = initial_filter(step=filter_step, aim_freq=lowpass_upper, sample_rate=sample_rate,
        #                         filter_type='lowpass')

        filteddata = tmp_to_filter_data.res_total.to_numpy()
        # signal.filtfilt(bk, ak, tmp_to_filter_data.res_total.to_numpy())
        print(filter_length)
        tmp_to_filter_data['res_smooth'] = pd.Series(list(filteddata))
        tmp_to_filter_data['res_stat'] = tmp_to_filter_data['res_total'] - tmp_to_filter_data['res_smooth']

        #         (x_data,win=30,history_len=1440,predict_len=60,sample_rate=1440,event_num=7,daliy_event_num=2,inter_step=5)
        #         train_his, train_labels, y_his, y_labels,prop_his,prop_labels
        train_his, train_labels, y_his, y_labels, prop_his, prop_labels = find_lowest_dtw2(x_data=tmp_to_filter_data,
                                                                                           win=dtw_win,
                                                                                           history_len=train_history_len,
                                                                                           predict_len=predict_len,
                                                                                           sample_rate=sample_rate,
                                                                                           event_num=event_num,
                                                                                           daliy_event_num=daliy_event_num,
                                                                                           dtw_length=dtw_length)

        test_results['history'].append(np.array(train_his))
        test_results['predict'].append(np.array(train_labels))

        y_test_results['history'].append(np.array(y_his))
        y_test_results['predict'].append(np.array(y_labels))

        prop_test_results['history'].append(np.array(prop_his))
        prop_test_results['predict'].append(np.array(prop_labels))
        # test_stat['history'].append(np.array(stat_his))
        # test_stat['predict'].append(np.array(stat_labels))

    train_results['history'] = np.array(train_results['history'])
    train_results['predict'] = np.array(train_results['predict'])

    # train_stat['history'] = np.array(train_stat['history'])
    # train_stat['predict'] = np.array(train_stat['predict'])
    y_results['history'] = np.array(y_results['history'])
    y_results['predict'] = np.array(y_results['predict'])

    prop_results['history'] = np.array(prop_results['history'])
    prop_results['predict'] = np.array(prop_results['predict'])

    test_results['history'] = np.array(test_results['history'])
    test_results['predict'] = np.array(test_results['predict'])

    y_test_results['history'] = np.array(y_test_results['history'])
    y_test_results['predict'] = np.array(y_test_results['predict'])

    prop_test_results['history'] = np.array(prop_test_results['history'])
    prop_test_results['predict'] = np.array(prop_test_results['predict'])

    # test_stat['history'] = np.array(test_stat['history'])
    # test_stat['predict'] = np.array(test_stat['predict'])
    # train_results, y_results,prop_results
    # train_results, y_results,prop_results, test_results, y_test_results,prop_test_results

    return train_results, y_results, prop_results, test_results, y_test_results, prop_test_results





def build_raw_resdiual_base(history_data, test_data, prop_model, win_prop=1440, min_cycle=14, start_id=1440 * 7,
                            end_id=-1,
                            up_step=1, sample_rate=1440, lowpass_upper=24, filter_step=3,
                            predict_len=60, dtw_win=30, train_history_len=1440, event_num=7, daliy_event_num=2):
    prop_res = raw_predict_qps_results(prop_model, future_step=win_prop, freq='min', except_train=False)

    time01 = time.time()
    raw_result = prop_res.copy().sort_values(['ds']).reset_index(drop=True)
    raw_result = raw_result[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]

    print(raw_result.shape)
    print(raw_result.dropna().shape)

    raw_predict = raw_result.copy().tail(win_prop).reset_index(drop=True)

    input_history = history_data.copy().sort_values(['ds']).reset_index(drop=True)

    res_train = raw_result.head(prop_res.shape[0] - win_prop).reset_index(drop=True)
    res_train['res_total'] = input_history['y']
    # - res_train['yhat']
    res_train['y'] = input_history.copy().reset_index(drop=True)['y']

    print(res_train.shape)
    print(res_train.dropna().shape)

    res_test = raw_predict.copy().sort_values(['ds']).reset_index(drop=True)

    input_test = test_data.copy().sort_values(['ds']).reset_index(drop=True)
    res_test['res_total'] = input_test['y']
    # - res_test['yhat']
    res_test['y'] = input_test.copy().reset_index(drop=True)['y']

    res_train = res_train.sort_values(['ds']).reset_index(drop=True)
    res_test = res_test.sort_values(['ds']).reset_index(drop=True)

    print(res_test.shape)
    print(res_test.dropna().shape)

    #     res_train = res_train.tail(res_train.shape[0]-start_id).sort_values(['ds']).reset_index(drop=True)

    prop_trains = {'history': [], 'predict': []}
    prop_test = {'history': [], 'predict': []}

    y_trains = {'history': [], 'predict': []}
    y_test = {'history': [], 'predict': []}

    train_results = {'history': [], 'predict': []}
    train_stat = {'history': [], 'predict': []}

    end_step = np.min([end_id, res_train.shape[0] - predict_len])

    if end_id < 0:
        end_step = res_train.shape[0] - predict_len

    for step in range(start_id, end_step, up_step):
        time11 = time.time()
        filter_length = np.min([step, min_cycle * sample_rate])
        if filter_length + dtw_win < step:
            filter_length = filter_length + dtw_win
        tmp_to_filter_data = res_train[step - filter_length:step + predict_len].copy()
        tmp_to_filter_data = tmp_to_filter_data.reset_index(drop=True)

        bk, ak = initial_filter(step=filter_step, aim_freq=lowpass_upper, sample_rate=sample_rate,
                                filter_type='lowpass')

        filteddata = signal.filtfilt(bk, ak, tmp_to_filter_data.res_total.to_numpy())
        print(tmp_to_filter_data.res_total.shape)
        print(tmp_to_filter_data.res_total.dropna().shape)
        print(filteddata)
        print(filter_length)
        tmp_to_filter_data['res_smooth'] = pd.Series(list(filteddata))
        tmp_to_filter_data['res_stat'] = tmp_to_filter_data['res_total'] - tmp_to_filter_data['res_smooth']
        #         res_trains,res_labels,stat_trains,stat_labels,prop_tmp_trains,prop_tmp_labels,y_tmp_trains,y_tmp_labels

        train_his, train_labels, stat_his, stat_labels, prop_his, prop_labels, y_his, y_labels = find_lowest_data_base(
            x_data=tmp_to_filter_data,
            history_len=train_history_len,
            predict_len=predict_len,
            sample_rate=sample_rate,
            event_num=event_num,
            daliy_event_num=daliy_event_num)

        train_results['history'].append(np.array(train_his))
        train_results['predict'].append(np.array(train_labels))
        train_stat['history'].append(np.array(stat_his))
        train_stat['predict'].append(np.array(stat_labels))

        y_trains['history'].append(np.array(y_his))
        y_trains['predict'].append(np.array(y_labels))

        prop_trains['history'].append(np.array(prop_his))
        prop_trains['predict'].append(np.array(prop_labels))

        print(time.time() - time11)

    test_results = {'history': [], 'predict': []}
    test_stat = {'history': [], 'predict': []}

    res_total_data = pd.concat([res_train.copy().reset_index(drop=True), res_test.copy().reset_index(drop=True)],
                               axis=0).reset_index(drop=True)
    res_total_data = res_total_data.sort_values(['ds']).reset_index(drop=True)

    if end_id > 0:
        train_results['history'] = np.array(train_results['history'])
        train_results['predict'] = np.array(train_results['predict'])

        train_stat['history'] = np.array(train_stat['history'])
        train_stat['predict'] = np.array(train_stat['predict'])

        y_trains['history'] = np.array(y_trains['history'])
        y_trains['predict'] = np.array(y_trains['predict'])

        prop_trains['history'] = np.array(prop_trains['history'])
        prop_trains['predict'] = np.array(prop_trains['predict'])

        return train_results, train_stat, y_trains, prop_trains

    for step in range(res_train.shape[0], res_total_data.shape[0] - predict_len, up_step):
        filter_length = np.min([step, min_cycle * sample_rate])
        if filter_length + dtw_win < step:
            filter_length = filter_length + dtw_win

        tmp_to_filter_data = res_total_data[step - filter_length:step + predict_len].copy()
        tmp_to_filter_data = tmp_to_filter_data.reset_index(drop=True)

        bk, ak = initial_filter(step=filter_step, aim_freq=lowpass_upper, sample_rate=sample_rate,
                                filter_type='lowpass')

        filteddata = signal.filtfilt(bk, ak, tmp_to_filter_data.res_total.to_numpy())
        print(filter_length)
        tmp_to_filter_data['res_smooth'] = pd.Series(list(filteddata))
        tmp_to_filter_data['res_stat'] = tmp_to_filter_data['res_total'] - tmp_to_filter_data['res_smooth']

        #         res_trains,res_labels,stat_trains,stat_labels,prop_tmp_trains,prop_tmp_labels,y_tmp_trains,y_tmp_labels

        train_his, train_labels, stat_his, stat_labels, prop_his, prop_labels, y_his, y_labels = find_lowest_data_base(
            x_data=tmp_to_filter_data,
            history_len=train_history_len,
            predict_len=predict_len,
            sample_rate=sample_rate,
            event_num=event_num,
            daliy_event_num=daliy_event_num)

        test_results['history'].append(np.array(train_his))
        test_results['predict'].append(np.array(train_labels))
        test_stat['history'].append(np.array(stat_his))
        test_stat['predict'].append(np.array(stat_labels))

        y_test['history'].append(np.array(y_his))
        y_test['predict'].append(np.array(y_labels))

        prop_test['history'].append(np.array(prop_his))
        prop_test['predict'].append(np.array(prop_labels))

    train_results['history'] = np.array(train_results['history'])
    train_results['predict'] = np.array(train_results['predict'])

    train_stat['history'] = np.array(train_stat['history'])
    train_stat['predict'] = np.array(train_stat['predict'])

    test_results['history'] = np.array(test_results['history'])
    test_results['predict'] = np.array(test_results['predict'])

    test_stat['history'] = np.array(test_stat['history'])
    test_stat['predict'] = np.array(test_stat['predict'])

    y_trains['history'] = np.array(y_trains['history'])
    y_trains['predict'] = np.array(y_trains['predict'])

    prop_trains['history'] = np.array(prop_trains['history'])
    prop_trains['predict'] = np.array(prop_trains['predict'])

    y_test['history'] = np.array(y_test['history'])
    y_test['predict'] = np.array(y_test['predict'])

    prop_test['history'] = np.array(prop_test['history'])
    prop_test['predict'] = np.array(prop_test['predict'])

    return train_results, train_stat, y_trains, prop_trains, test_results, test_stat, y_test, prop_test

def make_dataset(x_d,input_len):
    time_s1 = x_d[:, 0, :, 0]
    time_s2 = x_d[:, :, input_len, 0]
    data_s1 = x_d[:, :, :, 1]
    data_s2 = x_d[:, :, input_len:, 1]

    out_data1 = data_s1.transpose((0, 2, 1))
    out_time1 = np.expand_dims(time_s1, axis=2)
    out1 = np.concatenate([out_time1, out_data1], axis=2)

    out_data2 = data_s2
    out_time2 = np.expand_dims(time_s2, axis=2)
    out2 = np.concatenate([out_time2, out_data2], axis=2)

    print(time_s1.shape)
    print(time_s2.shape)
    print(data_s1.shape)

    print(out_data1.shape)
    print(out_time1.shape)
    print(out1.shape)

    print(data_s2.shape)

    print(out_data2.shape)
    print(out_time2.shape)
    print(out2.shape)

    return out1, out2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[KAE-Informer] Data Preprocess algorithm')

    # , required=True
    parser.add_argument('--data_path', type=str, default='./data/data_submit',
                        help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

    parser.add_argument("--app",type=str,default='app8',help='selected app')

    parser.add_argument("--test_len",type=int,default=1440,help='length of the total predict len')

    # n_changepoints=30, changepoint_range=0.9,
    #                                                 changepoint_prior_scale=0.01
    parser.add_argument("--ts_extractor_n_c",type=int,default=30,help='number of changepoints for ts extractor based on the Prophet')
    parser.add_argument("--ts_extractor_range",type=float,default=0.9,help='range of the changepoint appearing for ts extractor based on the Prophet')
    parser.add_argument("--ts_extractor_scale",type=float,default=0.01,help=' changepoint delta rate for ts extractor based on the Prophet')

    parser.add_argument("--proced_data_path",type=str,default='./data/data_proced',help='the dir to save the data processed the event search')
    parser.add_argument("--out_data_path",type=str,default='./data/data_train_test',help='the dir to save the train and test dataset')

    parser.add_argument("--freq",type=str,default='min')
    parser.add_argument('--event_days',type=int,default=14,help='length of historical days to find the events')
    parser.add_argument("--start_day",type=int,default=18,help='the day start to process the dataset')
    parser.add_argument('--end_day',type=int,default=-1,help='the day ends to process the dataset, -1 means to the end of the data')
    parser.add_argument('--up_step',type=int,default=1,help='the step to construct each item in the training and testing dataset')
    parser.add_argument('--sample_rate',type=int,default=1440,help='how many data points for 1 day,default=1440 for min freq')
    parser.add_argument('--r_h_filtering_freq',type=int,default=24,help='the frequency to filtering out the R_H,default is 24Hz which means the filtering out the minute-level components')
    parser.add_argument('--filter_order',type=int,default=3,help='the order of the low pass filter')
    parser.add_argument('--predict_len',type=int,default=60,help='the steps of Y_L which is predicted by the Informer-architecture at once')
    parser.add_argument('--dtw_win',type=int,default=15,help='the width of the search window for the event extraction')
    parser.add_argument('--input_len',type=int,default=1440,help='the steps of X_L which is inputed to the main Informer architecture')
    parser.add_argument('--event_num',type=int,default=7,help='number of extracted events')
    parser.add_argument('--daliy_event_limit',type=int,default=2,help='the upper limit of the events extracted from 1 day')
    parser.add_argument('--dtw_length',type=int,default=480,help='how many prior steps are used to input to dtw algorithm when searching the event')

    #                                                                             start_id=18 * 1440,
    #                                                                             end_id=-1, up_step=1, sample_rate=1440,
    #                                                                             lowpass_upper=24, filter_step=3,
    #                                                                             predict_len=60, dtw_win=15,
    #                                                                             train_history_len=1440, event_num=7,
    #                                                                             daliy_event_num=2, dtw_length=480


    args = parser.parse_args()

    # , required=True
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path, exist_ok=True)

    if not os.path.exists(args.proced_data_path):
        os.makedirs(args.proced_data_path,exist_ok=True)

    if not os.path.exists(args.out_data_path):
        os.makedirs(args.out_data_path,exist_ok=True)

    aim4 = [args.app]


    qps_predict_data = {}

    for k in aim4:
        qps_predict_data[str(k)] = pd.read_csv(os.path.join(args.data_path,"%s.csv" % k))

    for k in qps_predict_data.keys():
        print(k)
        print('ds' in qps_predict_data[k].columns)
        qps_predict_data[k]['ds'] = pd.to_datetime(qps_predict_data[k]['ds'])
        qps_predict_data[k] = qps_predict_data[k][['ds', 'y']].sort_values(['ds']).reset_index(drop=True)

    qps_predict_train_test = {}

    for k in qps_predict_data.keys():
        qps_predict_train_test[k] = {}
        qps_predict_train_test[k]['std'] = qps_predict_data[k].y.std()
        qps_predict_train_test[k]['mean'] = qps_predict_data[k].y.mean()
        qps_predict_train_test[k]['train'] = (qps_predict_data[k].copy().head(qps_predict_data[k].shape[0] - 1440))
        qps_predict_train_test[k]['test'] = (qps_predict_data[k].copy().tail(args.test_len))
        trian_data = qps_predict_train_test[k]['train'].copy()
        test_data = qps_predict_train_test[k]['test'].copy()

    raw_prophet_models = {}
    for k in qps_predict_data.keys():
        # parser.add_argument("--ts_extractor_n_c",type=int,default=30,help='number of changepoints for ts extractor based on the Prophet')
        #     parser.add_argument("--ts_extractor_range",type=float,default=0.9,help='range of the changepoint appearing for ts extractor based on the Prophet')
        #     parser.add_argument("--ts_extractor_scale",type=f
        raw_prophet_models[k] = initial_Prophet(n_changepoints=args.ts_extractor_n_c, changepoint_range=args.ts_extractor_range,
                                                changepoint_prior_scale=args.ts_extractor_scale)
        raw_prophet_models[k].fit(qps_predict_train_test[k]['train'])

    for k in qps_predict_train_test.keys():
        qps_predict_train_test[k]['raw_predict'] = raw_predict_qps_results(raw_prophet_models[k], args.test_len,freq=args.freq,
                                                                           except_train=False)
        prop_data = qps_predict_train_test[k]['raw_predict'].copy()
        qps_predict_train_test[k]['prop_predict'] = prop_data[['ds', 'yhat']].copy()
        #     *qps_predict_train_test[k]['std']+qps_predict_train_test[k]['mean']
        qps_predict_train_test[k]['prop_predict']['y'] = prop_data.yhat
        loss, lower_loss, upper_loss = predict_multiple_results_abs(qps_predict_train_test[k]['test'],
                                                                    qps_predict_train_test[k]['raw_predict'],
                                                                    mean=qps_predict_train_test[k]['mean'],
                                                                    std=qps_predict_train_test[k]['std'], tail=args.test_len)
        print(loss)
        print(qps_predict_train_test[k]['mean'])
        qps_predict_train_test[k]['raw_loss'] = compute_relative_error(loss, qps_predict_train_test[k]['mean'])
        qps_predict_train_test[k]['raw_lower_loss'] = compute_relative_error(lower_loss,
                                                                             qps_predict_train_test[k]['mean'])
        qps_predict_train_test[k]['raw_upper_loss'] = compute_relative_error(upper_loss,
                                                                             qps_predict_train_test[k]['mean'])

    for k in qps_predict_train_test.keys():
        print("%s: loss: %f, lower_loss: %f, upper_loss: %f" % (k, qps_predict_train_test[k]['raw_loss'],
                                                                qps_predict_train_test[k]['raw_lower_loss'],
                                                                qps_predict_train_test[k]['raw_upper_loss']))



    for aqa in qps_predict_train_test.keys():
        print("now process app: %s" % aqa)
        # train_results, y_results, prop_results,train_stat, test_results,y_test_results, prop_test_results ,test_stat
        if args.start_day < 7:
            start_day = 7
        else:
            start_day = args.start_day

        start_id = start_day * args.sample_rate

        if args.end_day < 0 or args.end_day > 19:
            end_id = -1
        else:
            end_id = args.end_day * args.sample_rate

        train_results, y_results, prop_results, train_stat, test_results, y_test_results, prop_test_results, test_stat = build_resdiual(qps_predict_train_test[aqa]['train'],
                                                                            qps_predict_train_test[aqa]['test'],
                                                                            prop_model=raw_prophet_models[aqa],
                                                                            win_prop=args.test_len, min_cycle=args.event_days,
                                                                            start_id=start_id,
                                                                            end_id=end_id, up_step=args.up_step, sample_rate=args.sample_rate,
                                                                            lowpass_upper=args.r_h_filtering_freq, filter_step=args.filter_order,
                                                                            predict_len=args.predict_len, dtw_win=args.dtw_win,
                                                                            train_history_len=args.input_len, event_num=args.event_num,
                                                                            daliy_event_num=args.daliy_event_limit, dtw_length=args.dtw_length)

        # print("a")
        #     'qps_hours_predict_data/aggrhost_train_data2.npy'
        if not os.path.exists((os.path.join(args.proced_data_path, aqa))):
            os.makedirs((os.path.join(args.proced_data_path, aqa)), exist_ok=True)

        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_train_data0.npy" % aqa), train_results['history'])
        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_train_label0.npy" % aqa), train_results['predict'])

        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_train_stat_data0.npy" % aqa), train_stat['history'])
        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_train_stat_label0.npy" % aqa), train_stat['predict'])

        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_train_prophet_data_base0.npy" % aqa),
                prop_results['history'])
        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_train_prophet_label_base0.npy" % aqa),
                prop_results['predict'])

        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_train_data_base0.npy" % aqa), y_results['history'])
        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_train_label_base0.npy" % aqa), y_results['predict'])

        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_test_data.npy" % aqa), test_results['history'])
        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_test_label.npy" % aqa), test_results['predict'])

        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_test_stat_data.npy" % aqa), test_stat['history'])
        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_test_stat_label.npy" % aqa), test_stat['predict'])

        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_test_prophet_data_base.npy" % aqa),
                prop_test_results['history'])
        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_test_prophet_label_base.npy" % aqa),
                prop_test_results['predict'])

        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_test_data_base.npy" % aqa), y_test_results['history'])
        np.save(os.path.join(os.path.join(args.proced_data_path, aqa), "%s_test_label_base.npy" % aqa), y_test_results['predict'])

        if not os.path.exists((os.path.join(args.out_data_path, aqa))):
            os.makedirs((os.path.join(args.out_data_path, aqa)), exist_ok=True)

        app_out_dir = (os.path.join(args.out_data_path, aqa))
        # train_results, y_results, prop_results, train_stat, test_results, y_test_results, prop_test_results, test_stat
        test_y_data_npy = y_test_results['history']
        print("test raw data shape:")
        print(test_y_data_npy.shape)
        # train_prophet_data_base0
        test_prophet_data_npy = prop_test_results['history']

        test_y_label_npy = y_test_results['predict']
        print("test raw label shape:")
        print(test_y_label_npy.shape)

        test_prophet_label_npy = prop_test_results['predict']

        test_x_label_npy = test_results['predict']

        test_x_data_npy = test_results['history']

        test_y_data_concat_data = np.concatenate([test_y_data_npy, test_y_label_npy], axis=2)
        test_prophet_data_concat_data = np.concatenate([test_prophet_data_npy, test_prophet_label_npy], axis=2)

        print("total y test data shape:")
        print(test_y_data_concat_data.shape)
        d_test1, d_test2 = make_dataset(test_y_data_concat_data,input_len=args.input_len)
        print("y test data ts shape:")
        print(d_test1.shape)
        print("y test data event shape:")
        print(d_test2.shape)


        np.save(os.path.join(app_out_dir, "%s_test_base_ts.npy" % aqa), d_test1)
        np.save(os.path.join(app_out_dir, "%s_test_base_event.npy" % aqa), d_test2)

        dp_test1, dp_test2 = make_dataset(test_prophet_data_concat_data,input_len=args.input_len)
        print("test prophet data ts shape:")
        print(dp_test1.shape)
        print("test prophet data event shape:")
        print(dp_test2.shape)

        np.save(os.path.join(app_out_dir, "%s_test_prophet_ts.npy" % aqa), dp_test1)
        np.save(os.path.join(app_out_dir, "%s_test_prophet_event.npy" % aqa), dp_test2)

        test_x_data_concat_data = np.concatenate([test_x_data_npy, test_x_label_npy], axis=2)

        dx_test1, dx_test2 = make_dataset(test_x_data_concat_data, input_len=args.input_len)
        print("x test data ts shape:")
        print(dx_test1.shape)
        print("x test data event shape:")
        print(dx_test2.shape)

        np.save(os.path.join(app_out_dir, "%s_test_ts.npy" % aqa), dx_test1)
        np.save(os.path.join(app_out_dir, "%s_test_event.npy" % aqa), dx_test2)

        test_stat_label_npy = test_stat['predict']

        test_stat_data_npy = test_stat['history']

        test_stat_data_concat_data = np.concatenate([test_stat_data_npy, test_stat_label_npy], axis=2)

        stat_dx_test1, stat_dx_test2 = make_dataset(test_stat_data_concat_data, input_len=args.input_len)
        print("high frequency residual test data ts shape:")
        print(stat_dx_test1.shape)
        print("high frequency residual test data event shape:")
        print(stat_dx_test2.shape)

        np.save(os.path.join(app_out_dir, "%s_stat_test_ts.npy" % aqa), stat_dx_test1)
        np.save(os.path.join(app_out_dir, "%s_stat_test_event.npy" % aqa), stat_dx_test2)
        # train_results, y_results, prop_results, train_stat
        train_y_data_npy = y_results['history']
        print("train raw data shape:")
        print(train_y_data_npy.shape)
        # train_prophet_data_base0
        train_prophet_data_npy = prop_results['history']

        train_y_label_npy = y_results['predict']
        print("train raw label shape:")
        print(train_y_label_npy.shape)

        train_prophet_label_npy = prop_results['predict']

        train_x_label_npy = train_results['predict']

        train_x_data_npy = train_results['history']

        train_y_data_concat_data = np.concatenate([train_y_data_npy, train_y_label_npy], axis=2)
        train_prophet_data_concat_data = np.concatenate([train_prophet_data_npy, train_prophet_label_npy], axis=2)

        print("total y train data shape:")
        print(train_y_data_concat_data.shape)
        d_train1, d_train2 = make_dataset(train_y_data_concat_data, input_len=args.input_len)
        print("y train data ts shape:")
        print(d_train1.shape)
        print("y train data event shape:")
        print(d_train2.shape)

        np.save(os.path.join(app_out_dir, "%s_train_base_ts.npy" % aqa), d_train1)
        np.save(os.path.join(app_out_dir, "%s_train_base_event.npy" % aqa), d_train2)

        dp_train1, dp_train2 = make_dataset(train_prophet_data_concat_data, input_len=args.input_len)
        print("train prophet data ts shape:")
        print(dp_train1.shape)
        print("train prophet data event shape:")
        print(dp_train2.shape)

        np.save(os.path.join(app_out_dir, "%s_train_prophet_ts.npy" % aqa), dp_train1)
        np.save(os.path.join(app_out_dir, "%s_train_prophet_event.npy" % aqa), dp_train2)

        train_x_data_concat_data = np.concatenate([train_x_data_npy, train_x_label_npy], axis=2)

        dx_train1, dx_train2 = make_dataset(train_x_data_concat_data, input_len=args.input_len)
        print("x train data ts shape:")
        print(dx_train1.shape)
        print("x train data event shape:")
        print(dx_train2.shape)

        np.save(os.path.join(app_out_dir, "%s_train_ts.npy" % aqa), dx_train1)
        np.save(os.path.join(app_out_dir, "%s_train_event.npy" % aqa), dx_train2)

        train_stat_label_npy = train_stat['predict']

        train_stat_data_npy = train_stat['history']

        train_stat_data_concat_data = np.concatenate([train_stat_data_npy, train_stat_label_npy], axis=2)

        stat_dx_train1, stat_dx_train2 = make_dataset(train_stat_data_concat_data, input_len=args.input_len)
        print("high frequency residual train data ts shape:")
        print(stat_dx_train1.shape)
        print("high frequency residual train data event shape:")
        print(stat_dx_train2.shape)

        np.save(os.path.join(app_out_dir, "%s_stat_train_ts.npy" % aqa), stat_dx_train1)
        np.save(os.path.join(app_out_dir, "%s_stat_train_event.npy" % aqa), stat_dx_train2)



