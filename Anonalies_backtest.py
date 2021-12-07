import os
import sys
import pandas as pd
import numpy as np
from multiprocessing import Pool
import time
sys.path.append('/home/lzy01/毕业设计/code/Anomalies')
from MatrixTools import *

# --- step 1 : 导入因子和股票收益率数据，从日频转为月频
#------- 导入数据
sdt = '20080101'
edt = '20210930'
factor_names = ['beta12', 'beta6', 'beta1'] + ['idvff1', 'idvff6', 'idvff12'] \
                + ['turn1_daily', 'turn6_daily', 'turn12_daily'] \
                + ['vturn1_daily', 'vturn6_daily', 'vturn12_daily'] \
                + ['abturn_daily'] \
                + ['dtv1_daily', 'dtv6_daily', 'dtv12_daily'] \
                + ['vdtv1_daily', 'vdtv6_daily', 'vdtv12_daily'] \
                + ['idsc1', 'idsc6', 'idsc12'] \
                + ['ts1', 'ts6', 'ts12'] \
                + ['mchg']\
                + ['imom11']\
                + ['rdmq']\
                + ['tanq']\
                + ['alaq']

eodData = EODMatrix()
factorData = EODMatrix(factor = True)

eod_mats = eodData.fetch_data(
           names = ['S_DQ_ADJCLOSE', 'FLOAT_A_SHR', 'adj_pct_chg'],
           sdt = sdt,
           edt = edt)
fac_mats = factorData.fetch_data(
           names = factor_names,
           sdt = sdt,
           edt = edt)

# ------- 频率转换
frechger = MatrixFreqTransformer()

eod_mat_M = frechger.freq_chg(
    data_dict = eod_mats,
    names = ['S_DQ_ADJCLOSE', 'FLOAT_A_SHR', 'adj_pct_chg'],
    funcs = [frechger.raw, frechger.raw, frechger.acc_ret]
)

fac_mat_M = frechger.freq_chg(
    data_dict = fac_mats,
    names = factor_names,
    funcs = [frechger.raw for i in factor_names]
)
#### 对比数据
cap = eod_mat_M['S_DQ_ADJCLOSE'] * eod_mat_M['FLOAT_A_SHR']
cap_filter = np.where(pd.DataFrame(cap).rank(axis=1, pct=True).values < 0.3, np.nan, 1)
cap_filter = np.where(np.isnan(cap), np.nan, cap_filter)
del(fac_mats)
del(eod_mats)

# --- step 1 结束

# --- step 2 : 提取收益率矩阵，市值矩阵，对单因子进行回测
## -------- 按照回测模块的格式加工
factor_input = {
    'DATE': fac_mat_M['date'],
    'CODE': fac_mat_M['code'],
    'DATA':{}
}
for name in factor_names:
    factor_input['DATA'][name] = fac_mat_M[name]

lagger = LagFactor(n_lag = 1)
factor_input = lagger.lag(factor_input)

ret_input = {
    'DATE':eod_mat_M['date'][1:],
    'CODE':eod_mat_M['code'],
    'RET':eod_mat_M['adj_pct_chg'][1:, :]
}

weight = np.log((pd.DataFrame(cap).shift(1).values)[1:, :])  #市值加权
rank_filter = (pd.DataFrame(cap_filter).shift(1).values)[1:, :]
# --- step 2 结束

# --- step 3 : 单因子回测
fig_path = '/home/lzy01/毕业设计/result/Anomalies/2008-2021/VW/fig'
sheet_path = '/home/lzy01/毕业设计/result/Anomalies/2008-2021/VW/sheet'
ranker_path = '/home/lzy01/毕业设计/result/Anomalies/2008-2021/VW/ranker'
factorRanker = SingleSorting(n_groups=10)

##### ------ 多线程单因子回测
def backtest(name):
    print(f'--------------- {name} backtest begin! ---------------')
    factorRanker.Analysis(
        factor_dict=factor_input,
        factor_name=name,
        rank_filter = rank_filter,
        weight = weight,
        ret_dict=ret_input,
        show = False,
        save = True,
        save_path = fig_path)
    factorRanker.group_ts.to_pickle(os.path.join(sheet_path, f'{name}_group_ts.pkl'))
    with open(os.path.join(ranker_path, f'{name}_ranker.pkl'), 'wb') as f:
        pickle.dump(factorRanker, f)
    print(f'--------------- {name} backtest finished! ---------------')
    return

t0 = time.time()
pool = Pool(20)
for name in factor_names:
    pool.apply_async(func = backtest, args = (name,), error_callback = lambda x:print(x))
pool.close()
pool.join()
print(f'--------------- total_time = {round(time.time() - t0)}s -------------')
# --- step 3 结束

# --- step 4 : 计算corr和mispricing
corrs = get_factor_corr(fac_mat_M, ['idvff6', 'beta12'])
fig = plt.figure(figsize= (12,9), dpi = 100)
plt.plot(list(range(len(corrs))), corrs, 'o-')
plt.xlabel('t', size = 15)
plt.ylabel('cross sectional corr', size = 15)
plt.show()
# plt.plot(list(range(len(corrs))), corrs)
# plt.show()


### 确定mispricing因子的成分
valid_factor = ['turn1_daily', 'vturn1_daily', 'abturn_daily', 'dtv1_daily', 'vdtv1_daily',
                'idsc1', 'ts1', 'mchg', 'rdmq', 'tanq', 'alaq']
# direction = [-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1]
direction = [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]
#imom的方向没法确定
### 计算mispricing因子
misprice = get_mispricing_score(factor_dict =  factor_input,
                                names = valid_factor,
                                directions = direction)
factorRanker.Analysis(
        factor_dict=misprice,
        factor_name='mispricing',
        rank_filter = rank_filter,
        weight = weight,
        ret_dict=ret_input,
        show = False,
        save = True,
        save_path = '/home/lzy01/毕业设计/data/anomalies/IVOL_timing')
# --- step 4 :结束

# --- step 5 : 用CH3调整收益率序列
adjustor = RiskAdjustor()
summary = {}
for name in tqdm(factor_names):
    # 读取对应因子的HML时间序列
    HML = pd.read_pickle(os.path.join(sheet_path, f'{name}_group_ts.pkl'))
    HML = HML.reset_index().rename(columns = {'index':'month'})[['month', 'HML']]
    alpha,t = adjustor.CH3Adjust(
        R = HML,
        t_index = 'month',
        r_index = 'HML'
    )
    summary[name] = [alpha, t]
summary = pd.DataFrame(summary).T


# --- step6 double sorting
multi_input = copy.deepcopy(misprice)
multi_input['DATA']['idvff'] = factor_input['DATA']['idvff12']
multi_input['DATA']['beta'] = factor_input['DATA']['beta12']
doubleSorting = MultipleSorting()
doubleSorting.multiple_rank(
    factor_dict = multi_input,
    factor_names = ['mispricing', 'idvff'],
    group_nums = [5, 10],
    filters = [rank_filter]
)
doubleSorting.multi_rank_analysis(
    ret_dict = ret_input,
    weight = weight
)

### --- 对于double sorting的结果进行alpha检验
sheet = pd.DataFrame(doubleSorting.group_ts)
sheet['month'] = doubleSorting.factor_dict['DATE']
col_index = ['group_' + (str(x).zfill(4)) + '_raw' for x in doubleSorting.group_codes_str]
alphas = np.zeros(tuple(doubleSorting.group_nums))
Ts = np.zeros(tuple(doubleSorting.group_nums))

for idx, code in tqdm(enumerate(doubleSorting.group_codes)):
    data = sheet[['month', col_index[idx]]]
    alpha, t = adjustor.CH3Adjust(
        R=data,
        t_index='month',
        r_index=col_index[idx]
    )
    alphas[tuple([x-1 for x in code])] = alpha
    Ts[tuple([x-1 for x in code])] = t

alphas = pd.DataFrame(alphas, index = [f'Overprice_{x}' for x in range(1,6)], columns = [f'IVOL_{x}'for x in range(1, 11)])

# 储存结果
# ----- IVOL
alphas.to_pickle('/home/lzy01/毕业设计/data/anomalies/IVOL_timing/IVOL12_5_10_alpha_VW.pkl')
IVOL_result = doubleSorting.group_result['exc_ret_mean']
IVOL_result = pd.DataFrame(IVOL_result, index = [f'Overprice_{x}' for x in range(1,6)], columns = [f'IVOL_{x}'for x in range(1, 11)])
IVOL_result.to_pickle('/home/lzy01/毕业设计/data/anomalies/IVOL_timing/IVOL12_5_10_abs_VW.pkl')
tvalue = pd.DataFrame(Ts, index = [f'Overprice_{x}' for x in range(1,6)], columns = [f'IVOL_{x}'for x in range(1, 11)])
tvalue.to_pickle('/home/lzy01/毕业设计/data/anomalies/IVOL_timing/IVOL12_5_10_tvalue_VW.pkl')

# ----- BETA
beta_result = doubleSorting.group_result['exc_ret_mean']
beta_result = pd.DataFrame(beta_result, index = [f'Overprice_{x}' for x in range(1,6)], columns = [f'BETA_{x}'for x in range(1, 11)])
beta_result.to_pickle('/home/lzy01/毕业设计/data/anomalies/IVOL_timing/BETA12_5_10_abs_EW.pkl')
tvalue = pd.DataFrame(Ts, index = [f'Overprice_{x}' for x in range(1,6)], columns = [f'BETA_{x}'for x in range(1, 11)])
tvalue.to_pickle('/home/lzy01/毕业设计/data/anomalies/IVOL_timing/BETA12_5_10_tvalue_EW.pkl')
# --- step6 finished
"""
idvff6， none weight，2008-2021,[5, 10]
beta12, none weight, 2008-2021, [5, 10]

单调性更好
idvff12, none weight, 2008-2021, [5,5]
idvff12, none weight, 2008-2021, [5,10]

beta12 & idvff12
"""


