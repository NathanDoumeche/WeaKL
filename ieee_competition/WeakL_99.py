# -*- coding:utf-8 -*-


from datetime import timedelta
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import time
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def is_running_on_gpu():
  if torch.cuda.is_available():
    print("The algorithm is running on GPU.")
  else:
    print("The algorithm is not running on GPU.")
is_running_on_gpu()

type_float64 = True

if type_float64:
  torch.set_default_dtype(torch.float64)
  np_dtype = np.float64
else:
  torch.set_default_dtype(torch.float32)
  np_dtype = np.float32


### Package WeakL

def normalize(dataset):
    data_min, data_max = np.min(dataset, axis=0), np.max(dataset, axis=0)
    center = (data_min+data_max)/2
    amplitude = data_max-data_min
    amplitude.replace(0, 1, inplace=True)
    return (dataset - center)/amplitude*2*torch.pi

def Sob_elementary(alpha, s, m):
    Sob_elem = torch.cat((torch.arange(-m, 0, device=device), torch.arange(1, m+1, device=device)))
    return alpha*(1+Sob_elem**(2*s))

def Sob_matrix(alpha_list, s_list, m_list):
    d = len(s_list)
    total_length = 1+sum(1 if m == 'Linear' else 2 * m for m in m_list)
    Sob_diag = torch.zeros(total_length, device=device)
    Sob_diag[0] = alpha_list[0]
    idx = 1
    for i in range(d):
        if m_list[i] == 'Linear':
            Sob_diag[idx] = alpha_list[i + 1]
            idx += 1
        else:
            Sob_diag[idx:idx + 2*m_list[i]] = Sob_elementary(alpha_list[i + 1], s_list[i], m_list[i])
            idx += 2*m_list[i]
    return torch.diag(Sob_diag)


def mat_frequency(m_list, n):
    total_length = 1+sum(2 * m if m != "Linear" else 1 for m in m_list)
    frequencies = torch.zeros(total_length, device=device)

    idx = 1
    for m in m_list:
        if m == "Linear":
            idx += 1
        else:
            frequencies[idx:idx + m] = torch.arange(-m, 0, device=device)
            frequencies[idx + m:idx + 2 * m] = torch.arange(1, m + 1, device=device)
            idx += 2 * m
    return torch.tile(frequencies, (n, 1))

def mat_linear(x_data, m_list, n):
    d = len(m_list)
    total_columns = 1+sum(1 if m == 'Linear' else 2 * m for m in m_list)
    mat = torch.zeros(n, total_columns, device=device)
    col_idx = 1
    for i in range(d):
        if m_list[i] == 'Linear':
            mat[:, col_idx] = x_data[:, i] - 1
            col_idx += 1
        else:
            col_idx += 2 * m_list[i]
    return mat

def mat_data(x_data, m_list, n):
    d = len(m_list)
    total_columns = 1+sum(1 if m == 'Linear' else 2 * m for m in m_list)
    mat = torch.zeros(n, total_columns, device=device)

    col_idx = 1
    for i in range(d):
        if m_list[i] == 'Linear':
            mat[:, col_idx] = x_data[:, i]
            col_idx += 1
        else:
            mat[:, col_idx:col_idx + 2 * m_list[i]] = torch.tile(x_data[:, i].view(-1, 1), (1, 2 * m_list[i]))
            col_idx += 2 * m_list[i]
    return mat


def phi_matrix(x_data, m_list):
    n = len(x_data)
    return torch.exp(-1j*torch.mul(mat_data(x_data, m_list, n),mat_frequency(m_list, n))/2)+mat_linear(x_data, m_list, n)

def regression_vector(x_data, y_data, m_list, alpha_list, s_list):
    covariance_matrix_X = torch.conj_physical(torch.transpose(phi_matrix(x_data, m_list), 0,1))@phi_matrix(x_data, m_list)
    covariance_XY = torch.conj_physical(torch.transpose(phi_matrix(x_data, m_list), 0,1))@y_data
    M = Sob_matrix(alpha_list, s_list, m_list)
    return torch.linalg.solve(covariance_matrix_X+len(x_data)*M, covariance_XY)

def estimation(fourier_coefs, z_data, m_list):
    fourier_coefs = fourier_coefs
    return phi_matrix(z_data, m_list)@fourier_coefs


def transform(df, m_list, fourier_vector, features):
    g_h = pd.DataFrame()
    g_h['Load'] = df['Load']
    g_h['Time']=df['Time']
    g_h['WeakL'] = torch.real(estimation(fourier_vector, torch.tensor(df[features].values, device=device), m_list)).view(-1).cpu().numpy()
    g_h["error"]=df["Load"]-g_h['WeakL']
    phi_mat = phi_matrix(torch.tensor(df[features].values, device=device), m_list)

    current = 1
    for j in range(len(features)):
        if m_list[j]=="Linear":
            linear_coeff = fourier_vector[current].cpu()
            g_h[features[j]] = linear_coeff*torch.tensor(df[features[j]].values).view(-1)
            current+= 1
        elif m_list[j] < 1:
            g_h[features[j]] = 0*1j
        else:
            partial_fourier = fourier_vector[current:current+2*m_list[j]]
            g_h[features[j]] = (phi_mat[:, current:current+2*m_list[j]]@partial_fourier).view(-1).cpu()
            current+= 2*m_list[j]
    return g_h

def calculate_total_length(m_list):
    return 1 + sum(2 * m if m != "Linear" else 1 for m in m_list)

def mat_frequency_h(m_list, n):
    total_length = calculate_total_length(m_list)
    frequencies = torch.zeros(total_length, device=device)

    idx = 1
    for m in m_list:
        if m == "Linear":
            idx += 1
        else:
            freq_range = torch.arange(1, m + 1, device=device)
            frequencies[idx:idx + m] = -freq_range.flip(0)
            frequencies[idx + m:idx + 2 * m] = freq_range
            idx += 2 * m
    return frequencies.unsqueeze(0).unsqueeze(0)

def mat_linear_h(x_data, m_list):
    batch_size, n, d = x_data.shape
    total_columns = calculate_total_length(m_list)
    mat = torch.zeros(batch_size, n, total_columns, device=device)

    col_idx = 1
    for i, m in enumerate(m_list):
        if m == 'Linear':
            mat[:, :, col_idx] = x_data[:, :, i] - 1
            col_idx += 1
        else:
            col_idx += 2 * m
    return mat

def mat_data_h(x_data, m_list):
    batch_size, n, d = x_data.shape
    total_columns = calculate_total_length(m_list)
    mat = torch.zeros(batch_size, n, total_columns, device=device)

    col_idx = 1
    for i, m in enumerate(m_list):
        if m == 'Linear':
            mat[:, :, col_idx] = x_data[:, :, i]
            col_idx += 1
        else:
            repeated_data = x_data[:, :, i].unsqueeze(2).expand(batch_size, n, 2 * m)
            mat[:, :, col_idx:col_idx + 2 * m] = repeated_data
            col_idx += 2 * m
    return mat

def phi_matrix_h(x_data, m_list):
    batch_size = x_data.size(0)
    n = x_data.size(1)
    return torch.exp(-1j * mat_data_h(x_data, m_list) * mat_frequency_h(m_list, n) / 2) + mat_linear_h(x_data, m_list)
def cov_hourly_m(m_list, data_hourly):
    cov_hourly = []
    x_data, x_test, y_data, ground_truth = data_hourly

    phi_mat = phi_matrix_h(x_data, m_list)
    covariance_matrix_X = torch.bmm(phi_mat.transpose(1, 2).conj(), phi_mat)
    covariance_XY = torch.bmm(phi_mat.transpose(1, 2).conj(), y_data)
    phi_mat_z = phi_matrix_h(x_test, m_list)

    return covariance_matrix_X, covariance_XY, phi_mat_z, ground_truth

def regression_vector_grid(covariance_matrix_X, covariance_XY, M):
    return torch.linalg.solve(covariance_matrix_X+M, covariance_XY)

def hour_formatting(data, date, features_weakl):
    begin_train, end_train, end_test = date["begin_train"], date["end_train"], date["end_test"]
    features, features1, features2 = features_weakl["features_union"], features_weakl["features1"], features_weakl["features2"]
    data_hourly = []

    for h in range(24):
        data_h = data[data['Hour']==h]

        data_h.loc[:,features]=normalize(data_h.loc[:,features]).loc[:,features]
        df_train = data_h[(data_h['Time']>= begin_train)&(data_h['Time']<end_train)]
        df_test = data_h[(data_h['Time']>= end_train)&(data_h['Time']<end_test)]

        if h<8:
            x_data = torch.tensor(df_train[features1].values, device=device)
            x_test = torch.tensor(df_test[features1].values, device=device)
        else :
            x_data = torch.tensor(df_train[features2].values, device=device)
            x_test = torch.tensor(df_test[features2].values, device=device)

        y_data = torch.tensor(df_train['Load'].values, device=device).view(-1,1)*(1+0*1j)
        ground_truth = torch.tensor(df_test['Load'].values, device=device)

        if type_float64:
          data_hourly.append([x_data, x_test, y_data, ground_truth])
        else:
          data_hourly.append([x_data.to(torch.float32), x_test.to(torch.float32), y_data.to(torch.complex64), ground_truth.to(torch.float32)])

    x_data = torch.stack([data_hourly[i][0] for i in range(24)])
    x_test = torch.stack([data_hourly[i][1] for i in range(24)])
    y_data = torch.stack([data_hourly[i][2] for i in range(24)])
    ground_truth = torch.stack([data_hourly[i][3] for i in range(24)])
    return x_data, x_test, y_data, ground_truth

def sob_effects(features_weakl, m_list, s_list, n):
    features_type = features_weakl["features_type"]
    n_param = len(features_type)+1
    one_list = [1. for i in range(n_param)]
    select_dim = [[0 for i in range(n_param)] for i in range(n_param)]
    for i in range(n_param):
        select_dim[i][i] = 1
    sobolev_effects = []
    for i in range(n_param):
        sobolev_effects.append(Sob_matrix(select_dim[i], s_list, m_list))
    sobolev_effects = torch.stack(sobolev_effects)*n
    return sobolev_effects

def WeakL(data, hyperparameters, cov_hourly, M_stacked):
    m_list, s_list, alpha_list = hyperparameters["m_list"], hyperparameters["s_list"], hyperparameters["alpha_list"]

    fourier_vectors= torch.linalg.solve(cov_hourly[0]+M_stacked, cov_hourly[1])
    estimators = torch.matmul(cov_hourly[2], fourier_vectors).squeeze(-1)
    mae_houlry = torch.mean(torch.abs(estimators-cov_hourly[3]), dim=1)
    mae = torch.mean(mae_houlry)
    return mae, fourier_vectors, mae_houlry

def create_grid(features_weakl, n, grid_parameters):
    features_type = features_weakl["masked"]
    grid_size_m, grid_size_p, grid_step_p = grid_parameters["grid_size_m"], grid_parameters["grid_size_p"], grid_parameters["grid_step_p"]
    number_regression = 0

    m_range, alpha_const, alpha_grid_range, s_list = [], [[10**(-30)]], [], [] # Initializing power with the offset
    for feature_type in features_type:
        if feature_type == "masked":
            m_range.append(["Linear"]), alpha_const.append([10**(10)]), s_list.append("*")
        elif feature_type in ["bool", "linear"]:
            m_range.append(["Linear"]), alpha_const.append([10**(-30)]), s_list.append("*")
        elif feature_type[:11] == "categorical":
            n_categories = int(feature_type[11:])
            m_range.append([n_categories//2+n_categories%2]), alpha_const.append([10**(-30)]), s_list.append(0)
        elif feature_type == "regression":
            m_ini = grid_parameters["m_ini"]
            power_ini = -int(np.log(n)/np.log(10))
            m_possibilities = list(range(m_ini-grid_size_m, m_ini+grid_size_m+1))

            power_possibilities = np.arange(power_ini-grid_size_p*grid_step_p, power_ini+(grid_size_p+1)*grid_step_p, grid_step_p)
            m_range.append(m_possibilities), alpha_grid_range.append(10**power_possibilities), s_list.append(2)
            number_regression+=1
    grid_m = list(itertools.product(*m_range))

    grid_a = torch.cartesian_prod(*torch.tensor(np.array(alpha_grid_range, dtype=np_dtype), device=device)).view(-1, number_regression, 1, 1)
    alpha_const = torch.tensor(alpha_const, device=device).view(-1, 1, 1)

    regression_mask = [i+1 for i in range(len(features_type)) if features_type[i] == "regression"]
    non_reg_mask = [0]+[i+1 for i in range(len(features_type)) if features_type[i] != "regression"]
    return grid_m, alpha_const, grid_a, s_list, regression_mask, non_reg_mask


def grid_search_weakl(data, date, features_weakl, n, grid_parameters):
    grid_m, alpha_const, grid_a, s_list, regression_mask, non_reg_mask = create_grid(features_weakl, n, grid_parameters)
    len_grid_m, len_grid_a, counter = len(grid_m), len(grid_a), 0

    grid_a = grid_a.split(grid_parameters["batch_size"], dim=0)
    batch_number = len(grid_a)

    mae_min = torch.inf
    mae_h=[]
    m_list_opt, power_list_opt, mae_list_opt, fourier_opt = [], [], [], []
    alpha_list_opt = torch.tensor([], device=device)

    data_hourly = hour_formatting(data, date, features_weakl)

    for m_list in grid_m:
      cov_hourly = cov_hourly_m(m_list, data_hourly)

      sobolev_effects = sob_effects(features_weakl, m_list, s_list, len(data_hourly[0][0]))
      mul1 = alpha_const*(sobolev_effects[non_reg_mask,:].unsqueeze(0))

      counter_batch =0
      print(str(counter*len_grid_a)+"/"+str(len_grid_m*len_grid_a))

      for grid_a_batch in grid_a:
        print("Batch: "+ str(counter_batch)+"/"+str(batch_number))
        counter_batch+=1

        mul2 = grid_a_batch*(sobolev_effects[regression_mask,:].unsqueeze(0))
        sobolev_matrices = torch.sum(mul1, dim=1, keepdim=True) + torch.sum(mul2, dim=1, keepdim=True)
        fourier_vectors= torch.linalg.solve(cov_hourly[0].unsqueeze(0)+sobolev_matrices, cov_hourly[1].unsqueeze(0))
        estimators = torch.matmul(cov_hourly[2].unsqueeze(0), fourier_vectors).squeeze(-1)
        errors = torch.abs(cov_hourly[3].unsqueeze(0)-estimators)
        mae_hourly =  torch.mean(errors, dim=2)
        mae_mean = torch.mean(mae_hourly, dim=1)
        min_mae_index = torch.argmin(mae_mean)

        if mae_mean[min_mae_index] < mae_min:
              m_list_opt, alpha_list_opt, mae_opt, fourier_opt, mae_h = m_list, grid_a_batch[min_mae_index], mae_mean[min_mae_index], fourier_vectors[min_mae_index], mae_hourly[min_mae_index]
              mae_min = mae_mean[min_mae_index]
      counter+=1
    alpha_opt = torch.zeros(len(regression_mask)+len(non_reg_mask), device=device)
    alpha_opt[regression_mask] = alpha_list_opt.view(-1)
    alpha_opt[non_reg_mask] = alpha_const.view(-1)
    return m_list_opt, alpha_opt, s_list, mae_opt, fourier_opt, mae_h

def calculate_total_length(m_list):
    return 1 + sum(2 * m if m != "Linear" else 1 for m in m_list)

def mat_frequency_online(m_list):
    total_length = calculate_total_length(m_list)
    frequencies = torch.zeros(total_length, device=device)

    idx = 1
    for m in m_list:
        if m == "Linear":
            idx += 1
        else:
            freq_range = torch.arange(1, m + 1, device=device)
            frequencies[idx:idx + m] = -freq_range.flip(0)
            frequencies[idx + m:idx + 2 * m] = freq_range
            idx += 2 * m

    return frequencies.unsqueeze(0).unsqueeze(0).unsqueeze(0)

def mat_linear_online(x_data, m_list):
    batch_size, steps, n, d = x_data.shape
    total_columns = calculate_total_length(m_list)
    mat = torch.zeros(batch_size, steps, n, total_columns, device=device)

    col_idx = 1
    for i, m in enumerate(m_list):
        if m == 'Linear':
            mat[:, :, :, col_idx] = x_data[:, :, :, -1] - 1
            col_idx += 1
        else:
            col_idx += 2 * m
    return mat

def mat_data_online(x_data, m_list):
    batch_size, steps, n, d = x_data.shape
    total_columns = calculate_total_length(m_list)
    mat = torch.zeros(batch_size, steps, n, total_columns, device=device)

    mat[:, :, :, 0] = 1

    col_idx = 1
    for i, m in enumerate(m_list):
        if m == 'Linear':
            mat[:, :, :, col_idx] = x_data[:, :, :, i]
            col_idx += 1
        else:
            repeated_data = x_data[:, :, :, i].unsqueeze(-1).expand(batch_size, steps, n, 2 * m)
            mat[:, :, :, col_idx:col_idx + 2 * m] = repeated_data
            col_idx += 2 * m
    return mat


def mat_time_online(x_data, m_list):
    batch_size, steps, n, d = x_data.shape
    total_columns = calculate_total_length(m_list)
    mat = torch.zeros(batch_size, steps, n, total_columns, device=device)

    col_idx = 1
    for i, m in enumerate(m_list):
        if m == 'Linear':
            mat[:, :, :, col_idx] = x_data[:, :, :, -1]
            col_idx += 1
        else:
            repeated_data = x_data[:, :, :, -1].unsqueeze(-1).expand(batch_size, steps, n, 2 * m)
            mat[:, :, :, col_idx:col_idx + 2 * m] = repeated_data
            col_idx += 2 * m
    return mat

def phi_matrix_online(x_data, m_list):
    return  mat_data_online(x_data, m_list) *(torch.exp(-1j * mat_time_online(x_data, m_list)* mat_frequency_online(m_list) / 2) + mat_linear_online(x_data, m_list))

def cov_hourly_online(m_list, data_hourly):
    cov_hourly = []
    x_data, x_test, y_data, ground_truth = data_hourly

    phi_mat = phi_matrix_online(x_data, m_list)
    covariance_matrix_X = torch.matmul(phi_mat.transpose(2, 3).conj(), phi_mat)
    covariance_XY = torch.matmul(phi_mat.transpose(2, 3).conj(), y_data)
    phi_mat_z = phi_matrix_online(x_test, m_list)

    return covariance_matrix_X, covariance_XY, phi_mat_z, ground_truth


def hour_formatting_online(data, date, features_weakl, hyperparameters):
    begin_train, end_train, end_test = date["begin_train"], date["end_train"], date["end_test"]
    features, features1, features2 = features_weakl["features_union"], features_weakl["features1"], features_weakl["features2"]
    fourier_vectors = hyperparameters["fourier_vectors"]
    m_list = hyperparameters["m_list"]
    window = hyperparameters["window"]
    data_hourly = []

    for h in range(24):
        x_online_list, x_test_list, y_online_list=[],[],[]

        data_h = data[data['Hour']==h]

        data_h.loc[:,features]=normalize(data_h.loc[:,features]).loc[:,features]

        if h<8:
            g_h=transform(data_h, m_list, fourier_vectors[h], features1)
            g_h.loc[:,features1]=normalize(g_h.loc[:,features1]).loc[:,features1]

        else:
            g_h=transform(data_h, m_list, fourier_vectors[h], features2)
            g_h.loc[:,features2]=normalize(g_h.loc[:,features2]).loc[:,features2]

        current_end_train = pd.to_datetime(end_train)
        current_end_test = current_end_train+timedelta(days=window)
        g_h['Time']=pd.to_datetime(g_h['Time'])

        learning_window = len(g_h[(g_h['Time']>=begin_train)&(g_h['Time']<current_end_train)])

        while current_end_train < pd.to_datetime(end_test):
            g_train = g_h[(g_h['Time']>=begin_train)&(g_h['Time']<current_end_train)]
            g_test = g_h[(g_h['Time']>=current_end_train)&(g_h['Time']<current_end_test)]
            if h<8:
                x_online = torch.tensor(g_train[features1].values, device=device)
                x_test_online = torch.tensor(g_test[features1].values, device=device)
            else:
                x_online = torch.tensor(g_train[features2].values, device=device)
                x_test_online = torch.tensor(g_test[features2].values, device=device)

            if x_test_online.shape[0] != 0:
                y_online = torch.tensor(g_train['error'].values, device=device).view(-1,1)*(1+0*1j)

                if not type_float64:
                    x_online, x_test_online, y_online = x_online.to(torch.float32), x_test_online.to(torch.float32), y_online.to(torch.complex64)
                x_online_list.append(x_online[-learning_window:,:])
                x_test_list.append(x_test_online)
                y_online_list.append(y_online[-learning_window:,:])


            current_end_train = current_end_train+timedelta(days=window)
            current_end_test = min(current_end_test+timedelta(days=window), pd.to_datetime(end_test))

        g_test = g_h[(g_h['Time']>=end_train)&(g_h['Time']<end_test)]
        ground_truth_online = torch.tensor(g_test['error'].values, device=device)

        if not type_float64:
              ground_truth_online = ground_truth_online.to(torch.float32)
        data_hourly.append([torch.stack(x_online_list), torch.stack(x_test_list), torch.stack(y_online_list), ground_truth_online])

    x_data = torch.stack([data_hourly[i][0] for i in range(24)])
    x_test = torch.stack([data_hourly[i][1] for i in range(24)])
    y_data = torch.stack([data_hourly[i][2] for i in range(24)])
    ground_truth = torch.stack([data_hourly[i][3] for i in range(24)])

    return x_data, x_test, y_data, ground_truth

def formatting_online(data, date, features_weakl, hyperparameters,h):
    
    begin_train, end_train, end_test = date["begin_train"], date["end_train"], date["end_test"]
    features, features1, features2 = features_weakl["features_union"], features_weakl["features1"], features_weakl["features2"]
    fourier_vectors = hyperparameters["fourier_vectors"]
    m_list = hyperparameters["m_list"]
    window = hyperparameters["window"]
    
    x_online_list, x_test_list, y_online_list=[],[],[]

    data_h = data[data['Hour']==h]

    data_h.loc[:,features]=normalize(data_h.loc[:,features]).loc[:,features]

    if h<8:
        g_h=transform(data_h, m_list, fourier_vectors[h], features1)
        g_h.loc[:,features1]=normalize(g_h.loc[:,features1]).loc[:,features1]

    else:
        g_h=transform(data_h, m_list, fourier_vectors[h], features2)
        g_h.loc[:,features2]=normalize(g_h.loc[:,features2]).loc[:,features2]

    current_end_train = pd.to_datetime(end_train)
    current_end_test = current_end_train+timedelta(days=window)
    g_h['Time']=pd.to_datetime(g_h['Time'])

    learning_window = len(g_h[(g_h['Time']>=begin_train)&(g_h['Time']<current_end_train)])

    while current_end_train < pd.to_datetime(end_test):
        g_train = g_h[(g_h['Time']>=begin_train)&(g_h['Time']<current_end_train)]
        g_test = g_h[(g_h['Time']>=current_end_train)&(g_h['Time']<current_end_test)]
        if h<8:
            x_online = torch.tensor(g_train[features1].values, device=device)
            x_test_online = torch.tensor(g_test[features1].values, device=device)
        else:
            x_online = torch.tensor(g_train[features2].values, device=device)
            x_test_online = torch.tensor(g_test[features2].values, device=device)

        if x_test_online.shape[0] != 0:
            y_online = torch.tensor(g_train['error'].values, device=device).view(-1,1)*(1+0*1j)

            if not type_float64:
                x_online, x_test_online, y_online = x_online.to(torch.float32), x_test_online.to(torch.float32), y_online.to(torch.complex64)
            x_online_list.append(x_online[-learning_window:,:])
            x_test_list.append(x_test_online)
            y_online_list.append(y_online[-learning_window:,:])


        current_end_train = current_end_train+timedelta(days=window)
        current_end_test = min(current_end_test+timedelta(days=window), pd.to_datetime(end_test))

    g_test = g_h[(g_h['Time']>=end_train)&(g_h['Time']<end_test)]
    ground_truth_online = torch.tensor(g_test['error'].values, device=device)

    if not type_float64:
            ground_truth_online = ground_truth_online.to(torch.float32)
    data_hourly.append([torch.stack(x_online_list), torch.stack(x_test_list), torch.stack(y_online_list), ground_truth_online])

    x_data = torch.stack(torch.stack(x_online_list))
    x_test = torch.stack(torch.stack(x_test_list))
    y_data = torch.stack(torch.stack(y_online_list))
    ground_truth = torch.stack([ground_truth_online])

    return x_data, x_test, y_data, ground_truth

def grid_search_online(data, date, features_weakl, n, grid_parameters, hyperparameters):
    grid_m, alpha_const, grid_a, s_list, regression_mask, non_reg_mask = create_grid(features_weakl, n, grid_parameters)
    len_grid_m, len_grid_a, counter = len(grid_m), len(grid_a), 0

    grid_a = grid_a.split(grid_parameters["batch_size"], dim=0)
    batch_number = len(grid_a)

    mae_min = torch.inf
    mae_h=[]
    m_list_opt, power_list_opt, mae_list_opt, fourier_opt = [], [], [], []
    alpha_list_opt = torch.tensor([], device=device)

    data_hourly = hour_formatting_online(data, date, features_weakl, hyperparameters)

    for m_list in grid_m:
      cov_hourly = cov_hourly_online(m_list, data_hourly)


      sobolev_effects = sob_effects(features_weakl, m_list, s_list, len(data_hourly[0][0][0]))
      mul1 = alpha_const*(sobolev_effects[non_reg_mask,:].unsqueeze(0))

      print(str(counter*len_grid_a)+"/"+str(len_grid_m*len_grid_a))
      counter_batch = 0
      for grid_a_batch in grid_a:
        print("Batch: "+str(counter_batch)+"/"+str(batch_number))
        counter_batch+=1

        mul2 = grid_a_batch*(sobolev_effects[regression_mask,:].unsqueeze(0))
        sobolev_matrices = torch.sum(mul1, dim=1, keepdim=True) + torch.sum(mul2, dim=1, keepdim=True)
        sobolev_matrices = sobolev_matrices.unsqueeze(2)

        fourier_vectors= torch.linalg.solve(cov_hourly[0].unsqueeze(0)+sobolev_matrices, cov_hourly[1].unsqueeze(0))
        estimators = torch.matmul(cov_hourly[2].unsqueeze(0), fourier_vectors).squeeze(-1)
        errors = torch.abs(cov_hourly[3].unsqueeze(0)-estimators.squeeze(-1))
        mae_hourly =  torch.mean(errors, dim=2)
        mae_mean = torch.mean(mae_hourly, dim=1)
        min_mae_index = torch.argmin(mae_mean)

        if mae_mean[min_mae_index] < mae_min:
              m_list_opt, alpha_list_opt, mae_opt, fourier_opt, mae_h = m_list, grid_a_batch[min_mae_index], mae_mean[min_mae_index], fourier_vectors[min_mae_index], mae_hourly[min_mae_index]
              mae_min = mae_mean[min_mae_index]
      counter+=1
    alpha_opt = torch.zeros(len(regression_mask)+len(non_reg_mask), device=device)
    alpha_opt[regression_mask] = alpha_list_opt.view(-1)
    alpha_opt[non_reg_mask] = alpha_const.view(-1)
    return m_list_opt, alpha_opt, s_list, mae_opt, fourier_opt, mae_h



##Test compet IEEE

data = pd.read_csv('data_corr.csv')
data['DayType']=np.float64(data.loc[:,'DayType'])
data['BH']=np.float64(data.loc[:,'BH'])
n = len(data['Time'])
data['time'] = [i/n*np.pi for i in range(n)]

# Remove Bank holidays
BH = np.array(np.where(data['BH']==1)).flatten()
sel = np.array((BH-24,BH,BH+24)).flatten()
data = data.drop(sel)



features_weakl = {"features1": ['FCloudCover_corr1','Load1D','Load1W','DayType','FTemperature_corr1','FWindDirection','FTemps95_corr1','Toy', 'time'],
"features2": ['FCloudCover_corr1','Load2D','Load1W','DayType','FTemperature_corr1','FWindDirection','FTemps95_corr1','Toy', 'time'],
"features_type": ["regression", "regression","regression",  "categorical7","linear","linear","linear","linear","linear"]}
features_weakl["features_union"] =  np.union1d(features_weakl["features1"],features_weakl["features2"])
features_weakl["masked"] = features_weakl["features_type"].copy()

print("\n Features 1 = ",features_weakl["features1"])
print("\n Features 2 = ",features_weakl["features2"])
print("\n TYPES :", features_weakl["features_type"])


for feature in features_weakl["features_union"]:
    data[feature]=np.float64(data.loc[:,feature])

dates_val = {"begin_train": "2017-03-18",
"end_train": "2019-10-01",
"end_test": "2020-01-01"}

print("\nGrid search offline #####################################")
grid_parameters ={
    "grid_size_m": 0,
    "m_ini": 3,
    "grid_size_p": 2,
    "grid_step_p": 0.5,
    "batch_size": 5*10**2
} 

n = len(data[(data['Time']>= dates_val["begin_train"])&(data['Time']<dates_val["end_train"])])//24

features_weakl_offline = copy.deepcopy(features_weakl)
print("\nValidation dates:")
print("Train: ",dates_val["begin_train"], " ---> ", dates_val["end_train"])
print("Test: ",dates_val["end_train"], " ---> ", dates_val["end_test"])

m_list, alpha_list, s_list, mae, fourier_vectors, mae_h = grid_search_weakl(data, dates_val, features_weakl_offline, n, grid_parameters)

hyperparameters = {"m_list": m_list,
                "s_list": s_list,
                "alpha_list": alpha_list,
                "fourier_vectors": fourier_vectors}
print("\nGrid search offline completed #####################################")
print("m_list = ",hyperparameters["m_list"])
print("alpha_list = ",hyperparameters["alpha_list"])
print("s_list = ",hyperparameters["s_list"])
print("Validation MAE = "+str(mae.cpu().numpy()))



print("\n\nTest offline #####################################")
dates_test = {"begin_train": "2017-03-18",
"end_train": "2020-01-01",
"end_test": "2020-03-01"}

print("\nOffline dates:")
print("Train: ",dates_test["begin_train"], " ---> ", dates_test["end_train"])
print("Test: ",dates_test["end_train"], " ---> ", dates_test["end_test"])
hyperparameters={"m_list":m_list,
                 "s_list":s_list,
                 "alpha_list":alpha_list}

data_hourly = hour_formatting(data, dates_test, features_weakl_offline)
cov_hourly = cov_hourly_m(m_list, data_hourly)
sobolev_matrix = Sob_matrix(alpha_list, s_list, m_list)*len(data_hourly[0][0])
M_stacked = torch.stack([sobolev_matrix for i in range(24)])
mae_test, fourier_vectors_test, mae_h_test = WeakL(data, hyperparameters, cov_hourly, M_stacked)

hyperparameters_test = {"m_list": m_list,
                   "s_list": s_list,
                   "alpha_list": alpha_list,
                   "fourier_vectors": fourier_vectors_test}

print("Test MAE = "+str(mae_test.cpu().numpy()))


hyperparameters = hyperparameters_test.copy()

print("\n\nGrid search Online #####################################")

hyperparameters["window"] = 1

dates_val_online = {"begin_train": "2020-03-01",
"end_train": "2020-11-18",
"end_test": "2021-01-18"}

print("\nValidation dates:")
print("Train: ",dates_val_online["begin_train"], " ---> ", dates_val_online["end_train"])
print("Test: ",dates_val_online["end_train"], " ---> ", dates_val_online["end_test"])

n = len(data[(data['Time']>= dates_val_online["begin_train"])&(data['Time']<dates_val_online["end_train"])])//24

features_weakl_online = copy.deepcopy(features_weakl)
features_weakl_online["features_type"] = ["regression" for i in range(len(features_weakl_online["features1"]))]
features_weakl_online["masked"] = features_weakl_online["features_type"].copy()

masked=[0]
for idx in masked:
    features_weakl_online["features_type"][idx] = "linear"
    features_weakl_online["masked"][idx] = "masked"

grid_parameters ={
    "grid_size_m": 0,
    "m_ini": 4,
    "grid_size_p": 1,
    "grid_step_p": 0.5,
    "batch_size": 1*10**2
}


m_online, alpha_online, s_online, mae_online_val, fourier_online_val, mae_list_opt = grid_search_online(data, dates_val_online, features_weakl_online, n, grid_parameters, hyperparameters)

print("\nGrid search online completed #####################################")
print("m_list online = ",m_online)
print("alpha_list online = ",alpha_online)
print("s_list online = ",s_online)
print("Validation MAE = ",mae_online_val.cpu().numpy())

dates_test_online = {"begin_train": "2020-05-01 ",
"end_train": "2021-01-18",
"end_test": "2021-02-17"}

print("\nOnline adaptation dates:")
print("Train: ",dates_test_online["begin_train"], " ---> ", dates_test_online["end_train"])
print("Test: ",dates_test_online["end_train"], " ---> ", dates_test_online["end_test"])
hyperparameters_test["window"] = 1

data_hourly = hour_formatting_online(data, dates_test_online, features_weakl, hyperparameters_test)
cov_hourly = cov_hourly_online(m_online, data_hourly)

sobolev_effects = sob_effects(features_weakl, m_online, s_online, len(data_hourly[0][0][0]))
sobolev_effects_online = (sobolev_effects*alpha_online.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

fourier_online_test= torch.linalg.solve(cov_hourly[0].unsqueeze(0)+sobolev_effects_online, cov_hourly[1].unsqueeze(0))
estimators = torch.matmul(cov_hourly[2].unsqueeze(0), fourier_online_test).squeeze(-1)
errors = torch.abs(cov_hourly[3].unsqueeze(0)-estimators.squeeze(-1))

WeakL_expert = pd.DataFrame()
WeakL_expert['Time']=data['Time']
WeakL_expert['Hour']=data['Hour']
WeakL_expert['Load']=data['Load']
WeakL_expert['WeakL']=np.zeros(len(data['Time']))

def predict_online(parameters_init, X, y, Q=0, sigma=1, delay=1):
    parameters = parameters_init
    d = X.shape[1]
    yhat = np.zeros(len(y))
    theta = parameters['theta']
    P = parameters['P']
    for t in range(len(y)):

        if t >= delay:
            P -= np.outer(P @ X[t-delay, :], P @ X[t-delay, :]) / (sigma**2 + (X[t-delay, :] @ P @ X[t-delay, :]).item())
            # theta = theta + P @ X[t-delay, :] * (y[t-delay] - (theta.T @ X[t-delay, :]).item()) / sigma**2

            theta += P @ X[t-delay, :].T * (y[t-delay] - X[t-delay, :]@theta) / sigma**2

            P += Q
        yhat[t] = (theta.T @ X[t, :]).item()
    return yhat
def intraday_corr(WeakL_expert, online):
  for h in range(24):
      mask = data['Hour']==h
      data_h=data[mask]
      if h<8:
          data_h.loc[:,features_weakl["features1"]]=normalize(data_h.loc[:,features_weakl["features1"]]).loc[:,features_weakl["features1"]]
          WeakL_expert.loc[mask,'WeakL']=transform(data_h, m_list, fourier_vectors_test[h], features_weakl["features1"]).loc[:,'WeakL']
      else:
          data_h.loc[:,features_weakl["features2"]]=normalize(data_h.loc[:,features_weakl["features2"]]).loc[:,features_weakl["features2"]]
          WeakL_expert.loc[mask,'WeakL']=transform(data_h, m_list, fourier_vectors_test[h], features_weakl["features2"]).loc[:,'WeakL']
      if online:
          tmp = WeakL_expert.loc[mask,'WeakL'].to_numpy()
          tmp[-estimators.shape[2]:]+=np.real(estimators[0,h,:].cpu().numpy().squeeze())
          WeakL_expert.loc[mask,'WeakL']=tmp

  WeakL_expert_corr = WeakL_expert.copy()
  res = (WeakL_expert_corr['Load']-WeakL_expert_corr['WeakL']).values
  X = np.zeros((WeakL_expert_corr.shape[0], 24))
  for t in range(WeakL_expert_corr.shape[0]):
      start = 24*max(0,(t-1)//24-2)+8
      end = start+24
      X[t,:]=res[start:end]
  d = X.shape[1]
  reshat_static = np.zeros(WeakL_expert_corr.shape[0])
  for h in range(24):
      sel = (WeakL_expert_corr['Hour']==h) & (WeakL_expert_corr['WeakL']!=0)
      reshat_static[sel]= predict_online({'theta':np.zeros(d), 'P':np.identity(d)}, X[sel,:].copy(), res[sel], delay = 1 if h<8 else 2)
  WeakL_expert_corr['WeakL']+=reshat_static

  if online:
    WeakL_expert_corr.to_csv('WeakL_expert_corr_online.csv')
  else:
    WeakL_expert_corr.to_csv('WeakL_expert_corr.csv')
  return WeakL_expert_corr


print("\n\nResults :")
online = False
WeakL_expert_corr = intraday_corr(WeakL_expert, online)
test = (WeakL_expert_corr['Time']>="2021-01-18") & (WeakL_expert_corr['Time']<"2021-02-17")
print("WeakL offline: ", np.mean(np.abs(WeakL_expert['Load'][test]-WeakL_expert['WeakL'][test])))
print("WeakL offline with intraday correction: ", np.mean(np.abs(WeakL_expert_corr['Load'][test]-WeakL_expert_corr['WeakL'][test])))

online = True
WeakL_expert_corr = intraday_corr(WeakL_expert, online)
test = (WeakL_expert_corr['Time']>="2021-01-18") & (WeakL_expert_corr['Time']<"2021-02-17")
print("WeakL online: ", np.mean(np.abs(WeakL_expert['Load'][test]-WeakL_expert['WeakL'][test])))
print("WeakL online with intraday correction: ", np.mean(np.abs(WeakL_expert_corr['Load'][test]-WeakL_expert_corr['WeakL'][test])))

Export = True
if Export:
    WeakL_expert_corr.to_csv('WeakL_mae99.csv')