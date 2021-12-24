import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import os
import time
import sys
import pickle
from tqdm import tqdm
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
import itertools
import statsmodels.api as sm

tqdm.pandas()

factor_path = '/home/lzy01/毕业设计/data/basic_db/factor_db.pkl'
EOD_path = '/home/lzy01/毕业设计/data/basic_db/eod_db.pkl'
CH3_path = '/home/lzy01/毕业设计/data/basic_db/CH3_M.pkl'


class Filter(object):
    __doc__ = """
    股票filter，用于计算一些交易filter，目前实现的有suspending ratio（过去1，6，12个月），IPO后180天, 市值后30%
    """

    def __init__(self, sdt, edt, suspand_periods=(22, 125, 252), suspend_ratio=0.25, ipo_lasting=180, cap_ratio=0.3):

        if isinstance(sdt, int) & isinstance(edt, int):
            sdt = str(sdt)
            edt = str(edt)

        self.sdt = pd.to_datetime(sdt)
        self.edt = pd.to_datetime(edt)
        self.suspend_periods = suspand_periods
        self.suspend_ratio = suspend_ratio
        self.ipo_lasting = ipo_lasting
        self.cap_ratio = cap_ratio
        self.EOD_path = EOD_path
        return

    @staticmethod
    def add_list(l):
        output = []
        for i in l:
            output += i
        return output

    def fetch_filter(self):
        """
        计算
        :return:
        """
        t0 = time.time()
        outpur_filter = {}

        # 从数据库中fetch data生产filter
        print('fetch data......')
        data_all = pd.read_pickle(self.EOD_path)
        data_all = data_all[(data_all['TRADE_DT'] >= self.sdt) & (data_all['TRADE_DT'] <= self.edt)] \
            [['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_VOLUME', 'S_DQ_ADJCLOSE', 'S_SHARE_TOTALA']]
        print(f'fetch data complete, time = {round(time.time() - t0, 2)}s')

        # 根据fetch的数据计算code和date, 生产对应的索引
        print('data preparation')
        codes = list(set(data_all['S_INFO_WINDCODE']))
        dates = list(set(data_all['TRADE_DT']))
        codes.sort()
        dates.sort()
        outpur_filter['date'] = dates
        outpur_filter['code'] = codes

        Index = pd.DataFrame(
            {
                'TRADE_DT': self.add_list([[x] * len(codes) for x in dates]),
                'S_INFO_WINDCODE': codes * len(dates)
            }
        )
        Index.sort_values(['TRADE_DT', 'S_INFO_WINDCODE'], ascending=True, inplace=True)
        data_all = pd.merge(Index, data_all,
                            on=['TRADE_DT', 'S_INFO_WINDCODE'],
                            how='left')
        data_all['cap'] = data_all['S_DQ_ADJCLOSE'] * data_all['S_SHARE_TOTALA']
        print('preparation finished!')

        print('cap filter')
        # filter 1: 市值后30%为nan，其余为1, 同时如果cap之前就为nan，filter中仍然为nan
        cap_matrix = data_all['cap'].values.reshape(len(dates), len(codes))
        cap_matrix_rank = pd.DataFrame(cap_matrix).rank(axis=1, pct=True).values
        cap_filter = np.where(cap_matrix_rank < self.cap_ratio, np.nan, 1)
        cap_filter = np.where(np.isnan(cap_matrix_rank), np.nan, cap_filter)
        outpur_filter['cap_filter'] = cap_filter
        outpur_filter['cap'] = cap_matrix
        print('cap filter finished!')

        print('suspension filter')
        # filter 2-4 : suspend days （在过去一个月，半年，一年suspend日期的比例不能超过一个阈值）
        vol_matrix = data_all['S_DQ_VOLUME'].values.reshape(len(dates), len(codes))
        self.vol_mat = vol_matrix
        for n_days in tqdm(self.suspend_periods):
            # 计算过去n天交易量累计不为nan的交易日数，以及交易量为0的交易日数，求二者的比
            trade_I = pd.DataFrame((~np.isnan(vol_matrix)).astype(np.int))
            susp_I = pd.DataFrame((vol_matrix == 0).astype(np.int))
            for n in range(1, 1 + n_days):
                trade_I += trade_I.shift(n, axis=1).fillna(0.0)
                susp_I += susp_I.shift(n, axis=1).fillna(0.0)
            susp_filter = np.where((susp_I / trade_I) >= self.suspend_ratio, np.nan, 1)

            # 前n天数据不足，填充为1
            susp_filter[:n_days, :] = 1
            # 原vol矩阵中为np.nan的数据在这里也填充为nan
            susp_filter[np.isnan(vol_matrix)] = np.nan
            # 储存输出
            outpur_filter[f'suspend_{n_days}days'] = susp_filter
        print('suspension filter finished!')

        print(f'filter processing finished, total time = {round(time.time() - t0, 2)}s')
        self.filters = outpur_filter

        return outpur_filter


class NewyWestTstats(BaseEstimator):
    __doc__ = """ Newy-West t Stats
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, axis=0, round=4) -> pd.DataFrame:
        """
        X : DataFrame, columns or indexs are the name of testing varible
        """
        if axis == 0:
            X = X.T
        names = X.index.to_list()
        X = np.array(X.values)
        summary = np.ones((4, len(X))) * np.nan

        for col in range(len(X)):
            x = X[col, :]
            x = x[~np.isnan(x)]
            ones = np.ones_like(x)
            results = sm.OLS(x, ones).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
            summary[0, col] = results.params[0]
            summary[1, col] = results.bse[0]
            summary[2, col] = results.tvalues[0]
            summary[3, col] = results.pvalues[0]

        self.summary = pd.DataFrame(summary.round(round), index=['mean', 'stderr', 't-value', 'p-value'], columns=names)

        return self.summary


class EODMatrix(object):
    __doc__ = """
    EOD数据矩阵类，用于生产EOD数据矩阵
    数据来源：
    1) '/root/pbcsf/Factor_Database/basic_data/Daily_return_with_cap.pkl'
    2) '/root/pbcsf/Factor_Database/output/daily_factors_all.pkl'
    """

    def __init__(self, factor=False):
        """
        初始化
        :param factor: 决定该实例是否是因子类，默认为false，即EOD数据类
        """
        self.path = EOD_path
        self.factor_path = factor_path
        self.if_factor = factor
        self.time_index = 'TRADE_DT'
        self.code_index = 'S_INFO_WINDCODE'
        self.data_format = 'Multi-Index'

        if factor:
            self.path = self.factor_path
        return

    @staticmethod
    def add_list(l):
        output = []
        for i in l:
            output += i
        return output

    def fetch_data(self, names: list, sdt: str, edt: str):
        """
        fetch data
        :param names:
        :param sdt:
        :param edt:
        :return:
        """
        # 加工原始数据，为矩阵生产准备
        t0 = time.time()

        print('start fetching data')
        data_all = pd.read_pickle(self.path)
        sdt, edt = pd.to_datetime(sdt), pd.to_datetime(edt)
        data_all = data_all[(data_all[self.time_index] >= sdt) & (data_all[self.time_index] <= edt)] \
            [[self.time_index, self.code_index] + names]
        codes = list(set(data_all[self.code_index]))
        dates = list(set(data_all[self.time_index]))
        codes.sort()
        dates.sort()
        self.dates = dates
        self.codes = codes
        Index = pd.DataFrame(
            {
                'TRADE_DT': self.add_list([[x] * len(codes) for x in dates]),
                'S_INFO_WINDCODE': codes * len(dates)
            }
        )
        Index.sort_values([self.time_index, self.code_index], ascending=True, inplace=True)
        data_all = pd.merge(Index, data_all,
                            on=['TRADE_DT', 'S_INFO_WINDCODE'],
                            how='left')
        print(f'fetching data complete, time = {round(time.time() - t0, 2)}s')

        print('matrix production')
        # 生产矩阵
        output_mats = {}
        for name in tqdm(names):
            output_mats[name] = data_all[name].values.reshape(len(dates), len(codes))

        output_mats['date'] = dates
        output_mats['code'] = codes
        print('matrix production complete')

        print(f'complete all, total time  = {round(time.time() - t0)}s')
        self.output_mats = output_mats

        return output_mats


class MatrixFreqTransformer(object):

    def __init__(self, basic_freq='d', to_freq='m'):
        self.basic_freq = basic_freq
        self.to_freq = to_freq
        return

    @staticmethod
    def get_month_end(dates):
        """
        获取输入序列每个月的最后一天,最终输出一个ndarray，第一列是年月str，第二列是对应日期在原始列表中的位置
        :param dates:
        :return:
        """
        dates.sort()
        Month = [str(x.year) + str(x.month).zfill(2) for x in dates]
        numm = []  # 每月最后一天的位置
        mon = []  # 月份
        for i in range(len(dates) - 1):
            '标记每月末'
            if Month[i] != Month[i + 1]:
                numm.append(i)
                mon.append(Month[i])
        # 不论最后一个是不是新月份，都要append
        numm.append(len(dates) - 1)
        mon.append(Month[len(dates) - 1])

        num = []
        num.append(mon)
        num.append(numm)
        num = np.array(num)
        numT = num.T
        numT[:, 1] = numT[:, 1].astype(np.int)
        return numT

    # TODO
    @staticmethod
    def get_quarter_end(dates):
        """
        获取输入序列每个季度的最后一天,最终输出一个ndarray，第一列是年季度str，第二列是对应日期在原始列表中的位置
        :param dates:
        :return:
        """

        def get_quarter(month):
            if month > 3 and month <= 6:
                return 2
            elif month > 6 and month <= 9:
                return 3
            elif month > 9:
                return 4
            else:
                return 1

        dates.sort()
        quarters = [str(x.year) + str(get_quarter(x.month)).zfill(2) for x in dates]
        numm = []  # 每月最后一天的位置
        qua = []  # 月份
        for i in range(len(dates) - 1):
            '标记每个季度的最后一天'
            if quarters[i] != quarters[i + 1]:
                numm.append(i)
                qua.append(quarters[i])
        # 不论最后一个是不是新月份，都要append
        numm.append(len(dates) - 1)
        qua.append(quarters[len(dates) - 1])

        num = []
        num.append(qua)
        num.append(numm)
        num = np.array(num)
        numT = num.T
        numT[:, 1] = numT[:, 1].astype(np.int)
        return numT

    def freq_chg(self, data_dict: dict, names: list, funcs: list, back_periods=None):
        """
        对于输入的变量矩阵，对其进行降频操作，日频降为月频，季频；即每个月，季度的最后一个交易日产出信号，信号
        由指定back period的数据以及指定的计算函数得出，最终输出新matrix，包含新索引以及数据矩阵
        :param names:
        :param freqs:
        :param funcs:
        :param back_periods:
        :return:
        """
        t0 = time.time()

        print('freq transform')

        dates = data_dict['date']
        dates.sort()
        codes = data_dict['code']

        # 根据需求进行转换
        if self.to_freq == 'm':
            range_end_and_idx = self.get_month_end(dates)
        elif self.to_freq == 'q':
            range_end_and_idx = self.get_quarter_end(dates)
        else:
            return
        self.end = range_end_and_idx  # 删除
        range_end = list(range_end_and_idx[:, 0])
        range_end_idx = [int(x) for x in list(range_end_and_idx[:, 1])]
        new_mats = {}

        print('transform complete')

        if back_periods is None:
            # 如果没有指定back period，就按照每一期的数据进行加工，与group by类似
            last_day_idx = [-1] + range_end_idx
            back_periods_per_range = [last_day_idx[i] - last_day_idx[i - 1] - 1 for i in range(1, len(last_day_idx))]
            self.delta = back_periods_per_range  # 删除
            back_periods = [9806 for i in names]

        print('fre change begins')
        # 根据back periods 和 funcs 计算得出新的matrix
        for name, func, back_period in tqdm(zip(names, funcs, back_periods)):
            mat = data_dict[name]  # 旧矩阵
            new_mat = np.ones((len(range_end_idx), len(codes))) * np.nan  # 新矩阵初始化
            for idx_new, x in enumerate(range_end_idx):  # 遍历新矩阵的每一个点，从旧矩阵对应的位置取出数据

                # 如果没有指定back period，则按照每个月的天数进行计算
                if back_period == 9806:
                    rolling_window = back_periods_per_range[idx_new]
                else:
                    rolling_window = back_period

                for y in range(len(codes)):
                    x = int(x)
                    array = mat[x - rolling_window:x + 1, y]
                    new_mat[idx_new, y] = func(array)

            # 储存新矩阵
            new_mats[name] = new_mat

        new_mats['code'] = codes
        new_mats['date'] = range_end  # YYYYMM
        print(f'freq change complete, total time ={round(time.time() - t0, 2)}s')

        return new_mats

    ## -- 一些常用的聚合函数
    @staticmethod
    def raw(x):
        # 保留原始因子
        return x[-1]

    @staticmethod
    def acc_ret(r):
        # 求累计收益率
        return np.cumprod(1 + r)[-1] - 1  # np.exp(np.nansum(np.log(r + 1))) - 1


class LagFactor(object):

    def __init__(self, n_lag=1):
        self.n = n_lag  # 滞后期数
        return

    def lag(self, factor_dict: dict, axis=0) -> dict:
        """
        对于输入的factor dict进行滞后n期的操作，factor dict为包含index，column，data的因子字典，
        默认行标签为时间，沿着axis=0进行lag，lag后即将前n期的nan drop，从DATE列表中删去对应数量的日期。
        :param factor_dict: {"DATE":list, "CODE":list, "DATA":np.nd-array[DATE, CODE]}
        :return: dict
        """
        output_dict = copy.deepcopy(factor_dict)

        # 对每一个因子进行lag操作
        shape = []
        for fac in output_dict['DATA']:
            lag_data = pd.DataFrame(output_dict['DATA'][fac]).shift(self.n, axis=axis).values
            if axis == 0:
                output_dict['DATA'][fac] = lag_data[self.n:, :]
            else:
                output_dict['DATA'][fac] = lag_data[:, self.n:]
            shape.append(output_dict['DATA'][fac].shape[axis])
        # 修改时间戳
        output_dict['DATE'] = output_dict['DATE'][self.n:]
        lenth = len(output_dict['DATE'])

        # 检查维度
        correct = sum([x == lenth for x in shape]) / len(shape)
        if correct < 1:
            print('dimensions do not correspond， check data')
        elif correct == 1:
            print('dimensions correspond !')

        return output_dict


class SingleSorting(object):

    def __init__(self, n_groups):
        self.n = n_groups
        return

    @staticmethod
    def cal_IC(
            X: np.ndarray,
            Y: np.ndarray
    ) -> np.ndarray:
        """
        计算行与行之间的IC值, nan robust
        :param X:
        :param Y:
        :return:
        """
        # 标准化
        x_ = X - np.nanmean(X, axis=1).reshape(-1, 1)
        y_ = Y - np.nanmean(Y, axis=1).reshape(-1, 1)

        # x,y任意一个为nan的位置，两个值都要为nan
        # cov
        x_cal = np.copy(x_)
        y_cal = np.copy(y_)
        x_cal[np.isnan(y_cal)] = np.nan
        y_cal[np.isnan(x_cal)] = np.nan

        x_cal[np.isnan(x_cal)] = 0.0
        y_cal[np.isnan(y_cal)] = 0.0

        cov = np.dot(x_cal, y_cal.T)

        # std
        std_x = np.nanstd(x_, axis=1).reshape(-1, 1) * np.ones_like(cov)
        std_y = np.nanstd(y_, axis=1).reshape(-1, 1) * np.ones_like(cov)
        std = np.dot(std_x, std_y.T)

        # corr
        n = np.sum(~np.isnan(x_cal), axis=1).reshape(-1, 1)
        corr = (cov / std) * (len(std) / n)

        return np.diag(corr)

    def factor_ranking(self, factor_dict: dict, factor_name: str, rank_filter=1, rank_method='dense'):
        """
        对于输入的单因子在rank_filter过滤后，在每个横截面上进行排序

        :param factor_dict: 输入的因子字典，包括code，date，data
        :param factor_name: 需要进行排序的单因子
        :param rank_filter: 因子排序时候的filter，排序之前会与因子矩阵相乘，不参与排序的位置为nan，其余为1，default为1
        :param rank_method: 排序使用方法，默认为dense
        :return:
        """
        self.factor_dict = factor_dict
        self.factor_name = factor_name
        raw_matrix = factor_dict['DATA'][factor_name]
        if isinstance(rank_filter, np.ndarray):
            if rank_filter.shape != raw_matrix.shape:
                raise NotImplementedError(
                    f'shape of filter is {rank_filter.shape}, which is not the same as {raw_matrix.shape},'
                    f'please check !')
        self.factor_metrix = raw_matrix * rank_filter

        # ranking
        self.factor_metrix_rank = (np.ceil(pd.DataFrame(self.factor_metrix)
                                           .rank(axis=1, pct=True, na_option='keep', method=rank_method)
                                           .mul(self.n))).values

        return

    def ranking_analysis(self, ret_dict: dict, weight=None):
        """
        通过已经完成的ranking对于因子进行分组研究，进行的计算包括：
        * 1）因子分组计算，计算每一组的加权收益率时间序列 : self.group_ts
        * 2）因子HML时间序列计算
        * 3）因子收益率时间序列的t值，均值: self.stats
        * 4）因子expanding IC: self.IC & self.RankIC
        * 5）因子IC时间序列: self.IC_ts


        :param ret_dict: 收益率矩阵dict，包括code，date，ret
        :param weight: 分组加权收益的权重（市值加权等）， 不用进行归一化，会自动进行归一化
        :param plot: 是否作图
        :param fig_path: 图片输出地址
        :return:
        """
        # TODO：
        # * Newey - west调整
        # * Fama - Mecbech
        t0 = time.time()

        print(' data prepare')
        # 检查ret的index和因子是否一致
        if (ret_dict['CODE'] != self.factor_dict['CODE']) | (ret_dict['DATE'] != self.factor_dict['DATE']):
            raise IndexError('ret index differs from factor index, can not compute !')

        # 将因子为nan的ret全部置为nan
        ret_matrix = ret_dict['RET']
        rank = self.factor_metrix_rank
        ret_matrix[np.isnan(rank)] = np.nan
        self.ret_matrix = ret_matrix

        # 检查weight，这里的weight是计算分组收益率时候的weight，比如市值，etc
        if weight is not None:
            if weight.shape != self.factor_metrix_rank.shape:
                raise IndexError(
                    f'weight shape ({weight.shape}) is not the same as factor shape({self.factor_metrix_rank.shape})')
            else:
                weight = weight * ((~np.isnan(rank)).astype(np.float64))
        else:
            weight = np.ones_like(rank) * ((~np.isnan(rank)).astype(np.float64))

        print(' grouping')
        # 根据分组和weight计算每一组的均值收益率, 累计均值收益率，超额平均收益，累计超额平均收益
        ret_mean = np.nansum(ret_matrix * weight, axis=1) / np.nansum(weight, axis=1)
        group_ts = {}
        group_acc_ts = {}
        for group in tqdm(range(1, self.n + 1)):
            filter = np.where(rank == float(group), 1.0, np.nan)
            group_weight = (weight * filter) / (np.nansum(weight * filter, axis=1).reshape(-1, 1))

            # 计算单期收益
            group_ts[f'group_{group}_raw'] = np.nansum(group_weight * ret_matrix, axis=1)
            group_ts[f'group_{group}_exc'] = group_ts[f'group_{group}_raw'] - ret_mean

            # 计算累计收益
            group_acc_ts[f'group_{group}_raw_acc'] = np.cumsum(group_ts[f'group_{group}_raw'])
            group_acc_ts[f'group_{group}_exc_acc'] = np.cumsum(group_ts[f'group_{group}_exc'])

        # 根据最高组和最低组符号计算HML
        delta = np.nanmean(group_ts[f'group_{self.n}_raw']) - np.nanmean(group_ts[f'group_1_raw'])
        group_ts['HML'] = group_ts[f'group_{self.n}_raw'] - group_ts[f'group_1_raw']
        if delta <= 0:
            group_ts['HML'] = -1 * group_ts['HML']
        group_acc_ts['HML_acc'] = np.cumsum(group_ts['HML'])
        self.group_ts = pd.DataFrame(group_ts, index=self.factor_dict['DATE'])
        self.group_acc_ts = pd.DataFrame(group_acc_ts, index=self.factor_dict['DATE'])

        print(' stats calculating')
        # 计算收益率序列的t统计量
        ts_mat = self.group_ts[[f'group_{i}_exc' for i in range(1, self.n + 1)]].values
        mean = np.zeros((1, ts_mat.shape[1]))
        t, p = ttest_1samp(a=ts_mat, popmean=mean, axis=0, nan_policy='omit')
        NWtest = NewyWestTstats().fit(X=ts_mat, axis=0).summary
        self.stats = pd.DataFrame()
        self.stats['t-value'] = list(t)[0]
        self.stats['p-value'] = list(p)[0]
        self.stats['NW-tvalue'] = NWtest.loc['t-value', :].values
        self.stats['NW-pvalue'] = NWtest.loc['p-value', :].values
        self.stats.index = [f'group_{i}_exc' for i in range(1, self.n + 1)]

        print(' IC calculating')
        # 计算IC的时间序列以及expanding IC
        # -- 计算expanding IC
        factor_1d = self.factor_metrix.reshape(-1, 1)
        factor_rank_1d = self.factor_metrix_rank.reshape(-1, 1)
        nan_idx = np.isnan(factor_1d)
        factor_1d = pd.Series(factor_1d[~nan_idx])
        factor_rank_1d = pd.Series(factor_rank_1d[~nan_idx])
        ret_1d = pd.Series(ret_matrix.reshape(-1, 1)[~nan_idx])

        self.IC = factor_1d.corr(ret_1d)
        self.RankIC = factor_rank_1d.corr(ret_1d)

        # -- 计算IC序列
        self.IC_ts = pd.DataFrame()
        self.IC_ts['IC'] = self.cal_IC(self.factor_metrix, self.ret_matrix)
        self.IC_ts.index = self.factor_dict['DATE']
        self.IC_ts['IC_acc'] = np.cumsum(self.IC_ts['IC'])

        # 分组计算结果总结
        self.stats['mean_return_exc'] = np.nanmean(
            self.group_ts[[f'group_{group}_exc' for group in range(1, self.n + 1)]],
            axis=0)
        self.stats = self.stats[['mean_return_exc', 't-value', 'p-value', 'NW-tvalue', 'NW-pvalue']]

        print(f' analysis finished! total time = {round(time.time() - t0, 2)} s')
        return

    def plot(self, show=True, save=False, fig_path=None):
        """
        对于结果进行作图分析，将图片储存在fig path下，名称为：因子_analysis_YYMMDDMMSS.png
        :param fig_path:
        :return:
        """
        t0 = time.time()
        print(f'start plot for {self.factor_name}')

        fig = plt.figure(figsize=(20, 50), dpi=100)
        fig.set_facecolor('#FFFFFF')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        spec = gridspec.GridSpec(ncols=2, nrows=5)
        x = list(range(len(self.factor_dict['DATE'])))

        ## fig1 : 分组折线图， 第一行
        ax = fig.add_subplot(spec[0, :])
        leg = []
        for group in range(1, self.n + 1):
            plt.plot(x, self.group_acc_ts[f'group_{group}_exc_acc'])
            leg.append(f'group_{group}')
        plt.legend(leg)
        ax.set_title('group acc excess return')

        ## fig2 : HML cumsum
        ax = fig.add_subplot(spec[1, :])
        plt.plot(x, self.group_acc_ts[f'HML_acc'])
        plt.legend(['HML_acc'])
        ax.set_title('HML acc return')

        ## fig3a：HML 分布
        ax = fig.add_subplot(spec[2, 0])
        plt.hist(self.group_ts['HML'], bins=int(len(x) / 3), alpha=0.5, color='purple', density=True)
        ax.set_title('HML Distribution')

        ## fig3b: IC分布
        ax = fig.add_subplot(spec[2, 1])
        plt.hist(self.IC_ts['IC'], bins=int(len(x) / 3), alpha=0.5, color='purple', density=True)
        ax.set_title('IC Distribution')

        ## fig4 : 累计IC
        ax = fig.add_subplot(spec[3, :])
        plt.plot(x, self.IC_ts['IC_acc'])
        ax.set_title('IC acc')

        ## fig5a：分组bar
        ax = fig.add_subplot(spec[4, 0])
        plt.bar([i for i in range(1, self.n + 1)], self.stats['mean_return_exc'])
        ax.set_title('Group mean raw ret')

        ## fig5b: mean-std fig
        # 对每只票计算因子的mean和std，观察分布
        fac_mat = self.factor_metrix
        fac_mat = fac_mat[:, (np.sum(np.isnan(fac_mat), axis=0) / fac_mat.shape[1]) < (2 / 3)]  # 对于因子值2/3以上都为nan的票删除
        mean = np.nanmean(fac_mat, axis=0).reshape(-1)
        std = np.nanstd(fac_mat, axis=0).reshape(-1)
        del_idx = np.abs(mean / std) > 10000
        mean = mean[~del_idx]
        std = std[~del_idx]

        ax = fig.add_subplot(spec[4, 1])
        cmap = plt.get_cmap('bwr')
        plt.scatter(mean, std, c=mean / (std + 1e-4), cmap=cmap, s=5)
        plt.colorbar()
        plt.legend(['mean / std of single stock'])
        plt.xlabel('mean')
        plt.ylabel('std')
        ax.set_title('mean-std fig')

        ## -- 储存
        plt.suptitle(
            f'{self.factor_name} analysis figure, {self.factor_dict["DATE"][0]}-{self.factor_dict["DATE"][-1]}',
            size=25)
        if show:
            plt.show()
        if save:
            plt.savefig(os.path.join(fig_path,
                                     self.factor_name + f'_analysis_{time.strftime("%Y%m%d%H:%M:%S", time.localtime())}.png'))
        plt.close()

        print(f'plot finished, total time = {round(time.time() - t0, 2)}s')

        return

    def Analysis(self,
                 factor_dict: dict, factor_name: str,
                 ret_dict: dict,
                 rank_filter=1, weight=None, rank_method='dense',
                 plot=True, show=True, save=False, save_path=None):

        print(f'{factor_name} analysis begin:')
        t0 = time.time()

        print('step 1 : factor ranking')
        self.factor_ranking(factor_dict=factor_dict, rank_filter=rank_filter, factor_name=factor_name,
                            rank_method=rank_method)
        print(f'factor ranking finished, total time = {time.time() - t0}s')

        t1 = time.time()
        print('step 2 : group analysis')
        self.ranking_analysis(ret_dict=ret_dict, weight=weight)
        print(f'group analysis finished, total time = {time.time() - t1}s')

        if plot:
            t2 = time.time()
            print('step 3 : plot fig')
            self.plot(show=show, save=save, fig_path=save_path)
            print(f'plot finished, total time = {time.time() - t2}s')

        print(f'{self.factor_name} analysis finished, total time = {time.time() - t0}s')
        return


class MultipleSorting(object):
    """
    在SingleSorting的基础上实现，实现对于输入多个因子的分组检验，sorting方式为independent sorting
    一般而言为2两个因子的double sorting，在两个因子的情境下，可以进行可视化
    """

    def __init__(self):
        self.sorting_method = 'Independent Sorting'
        try:
            example = SingleSorting(n_groups=1)
            del (example)
        except Exception as e:
            raise NotImplementedError('please initiate SingleSorting class !')

        return

    @staticmethod
    def _list_sum(l):
        output = ''
        for idx, i in enumerate(l):
            if idx != len(l) - 1:
                output += (str(i).zfill(2))
            else:
                output += str(i).zfill(2)
        return output

    def multiple_rank(
            self,
            factor_dict: dict,
            factor_names: list,
            group_nums: list,
            filters=None
    ) -> None:

        """
        多重排序函数，输入因子字典和进行多重排序的因子名称，以及每个因子的对应分组数，对于每个因子进行独立的分组，然后对
        这些分组进行交叉编号，产出一个rank matrix
        :param factor_dict:
        :param factor_names:
        :param group_nums:
        :return:
        """
        t0 = time.time()
        self.factor_names = factor_names
        self.factor_dict = factor_dict
        self.group_nums = group_nums

        # 对于filter进行检查
        if filters is None:
            filters = [1 for i in factor_names]
        else:
            if (len(filters) != len(factor_names)) & len(filters) != 1:
                raise NotImplementedError('filter missing, please check')
            elif len(filters) == 1:  # 如只输入了一个filter，则对于全部都复制这个filter
                filters = [filters[0] for i in factor_names]

        print('single sorting')
        # 对于每个因子，调用单因子rank，将排序的矩阵储存在一个跟输入因子顺序相同的列表中
        rank_matrixs = []
        rankers = []
        for idx, factor_name in tqdm(enumerate(factor_names)):
            ranker = SingleSorting(n_groups=self.group_nums[idx])
            ranker.factor_ranking(
                factor_dict=factor_dict,
                factor_name=factor_name,
                rank_filter=filters[idx]
            )
            rank_matrix_ = ranker.factor_metrix_rank

            # 将计算好的rank matrix以及对应的ranker储存在类别属性中，方便之后的调用
            # 其中储存rankers主要是以防要对单因子进行rank
            rank_matrixs.append(rank_matrix_)
            rankers.append(ranker)
        self.rank_matrixs = rank_matrixs
        self.rankers = rankers
        print('single sorting finished')

        print('encoding group info')
        # 按照产生的因子分组结果，对于每只股票进行编码
        # 先产生编码，然后按照编码进行分组确认
        group_codes = list(range(1, self.group_nums[0] + 1))
        for num in self.group_nums[1:]:
            group_codes = list(itertools.product(group_codes, list(range(1, num + 1))))
        self.group_codes = group_codes
        self.group_codes_str = [int(self._list_sum(code)) for code in self.group_codes]

        # 对于编码，进行分组的筛选：
        group_filters = {}
        group_matrix = (np.ones_like(self.rank_matrixs[0]) * np.nan)

        for group_id, group in tqdm(enumerate(group_codes)):
            sub_group_filters = []

            # 对于改组的code，逐一计算filter，保存在一个列表中
            for idx, sub_group in enumerate(group):
                sub_rank_filter = (rank_matrixs[idx] == sub_group)
                sub_group_filters.append(sub_rank_filter)

            # 对于所有的filter取交集
            group_filter = np.ones_like(sub_group_filters[0])
            for filter in sub_group_filters:
                group_filter = filter.astype(np.int) * group_filter
            group_filter = group_filter.astype(np.bool)

            # 记录filter
            group_filters[self.group_codes_str[group_id]] = group_filter

            # 对于符合要求的股票分组
            group_matrix[group_filter] = self.group_codes_str[group_id]

        print(f'group encoding finished, total time = {round(time.time() - t0, 2)}s')

        self.group_filters = group_filters
        self.group_mat = group_matrix

        return

    def multi_rank_analysis(self, ret_dict, weight=None):
        """
        针对已经计算完成的多分组矩阵以及收益率矩阵进行分析，计算分组信息
        :param ret_dict: 收益率字典，包括对应的收益率矩阵以及code，date
        :param weight: 计算平均收益率时的权重
        :return:

        """

        t0 = time.time()
        self.ret_dict = ret_dict

        print('data prepare')
        # 检查ret的index和因子是否一致
        if (ret_dict['CODE'] != self.factor_dict['CODE']) | (ret_dict['DATE'] != self.factor_dict['DATE']):
            raise IndexError('ret index differs from factor index, can not compute !')

        # 将因子为nan的ret全部置为nan
        ret_matrix = ret_dict['RET']
        rank = self.group_mat
        ret_matrix[np.isnan(rank)] = np.nan
        self.ret_matrix = ret_matrix

        # 检查weight，这里的weight是计算分组收益率时候的weight，比如市值，etc
        if weight is not None:
            if weight.shape != rank.shape:
                raise IndexError(
                    f'weight shape ({weight.shape}) is not the same as factor shape({self.factor_metrix_rank.shape})')
            else:
                weight = weight * ((~np.isnan(rank)).astype(np.float64))
        else:
            weight = np.ones_like(rank) * ((~np.isnan(rank)).astype(np.float64))

        print('grouping')
        # 利用以及计算的group_matrix以及group code str进行分组的收益率计算
        ret_mean = np.nansum(ret_matrix * weight, axis=1) / np.nansum(weight, axis=1)
        group_ts = {}
        group_acc_ts = {}

        ## 将各个分组的平均收益，平均超额收益，t-value，p-value分别储存在一个n维array中，索引分别为不同因子的分组
        group_result = {
            'raw_ret_mean': np.ones(tuple(self.group_nums)) * np.nan,
            'exc_ret_mean': np.ones(tuple(self.group_nums)) * np.nan,
            'p-value': np.ones(tuple(self.group_nums)) * np.nan,  # for excess
            't-value': np.ones(tuple(self.group_nums)) * np.nan,  # for excess
        }
        # 标注每个array的轴
        for i, name in enumerate(self.factor_names):
            group_result[f'axis{i}'] = name

        for group_code, group_num in tqdm(zip(self.group_codes_str, self.group_codes)):

            filter = np.where(rank == group_code, 1, np.nan)
            if np.sum(~np.isnan(filter)) == 0:
                # 如果该组没有数据，跳过运算
                continue

            group_weight = (weight * filter) / (np.nansum(weight * filter, axis=1).reshape(-1, 1))

            # 计算单期收益
            digit = 2 * len(self.factor_names)  # 编码的位数，每个因子两位
            str_code = str(group_code).zfill(digit)
            group_ts[f'group_{str_code}_raw'] = np.nansum(group_weight * ret_matrix, axis=1)
            group_ts[f'group_{str_code}_exc'] = group_ts[f'group_{str_code}_raw'] - ret_mean

            # 计算累计收益
            group_acc_ts[f'group_{str_code}_raw_acc'] = np.cumsum(group_ts[f'group_{str_code}_raw'])
            group_acc_ts[f'group_{str_code}_exc_acc'] = np.cumsum(group_ts[f'group_{str_code}_exc'])

            # 计算统计量进行多维矩阵的填充
            idx = tuple([code - 1 for code in group_num])
            group_result['raw_ret_mean'][idx] = np.nanmean(group_ts[f'group_{str_code}_raw'])
            group_result['exc_ret_mean'][idx] = np.nanmean(group_ts[f'group_{str_code}_exc'])

            t, p = ttest_1samp(group_ts[f'group_{str_code}_exc'], 0)
            group_result['p-value'][idx] = p
            group_result['t-value'][idx] = t

        self.group_result = group_result
        self.group_ts = group_ts
        self.group_acc_ts = group_acc_ts
        print(f'multi-group analysis finished! total time = {round(time.time() - t0, 2)}s')

        return

    def single_sorting(self, factor_name, ret_dict=None, rank_filter=None, weight=None, fig_path=None):
        """
        此函数为调用单因子评价函数进行计算。只需要提供单因子的名称
        需要在调用过multiple_rank()后才能使用
        :param factor_name:
        :return:
        """
        factor_idx = self.factor_names.index(factor_name)

        if ret_dict == None:
            try:
                ret_dict = self.ret_dict
            except Exception as e:
                print(e)
                raise NotImplementedError('no return dict, please .run multi_rank_analysis() or submit one')

        if rank_filter == None:
            try:
                rank_filter = self.group_filters[factor_idx]
            except Exception as e:
                print(e)
                rank_filter = 1

        if fig_path is not None:
            save = True
        else:
            save = False

        ranker = self.rankers[factor_idx]
        ranker.Analysis(
            factor_dict=self.factor_dict,
            factor_name=factor_name,
            ret_dict=ret_dict,
            rank_filter=rank_filter,
            weight=weight,
            plot=True, show=True, save=save, save_path=fig_path
        )

        return ranker


class RiskAdjustor(object):

    def __init__(self, model='CH3', freq='M'):

        self.data_path = '/home/lzy01/毕业设计/data/basic_db'

        self.model_name = model + '_' + freq + '.pkl'

        if self.model_name == 'CH3_M.pkl':
            self.CH3 = pd.read_pickle(os.path.join(self.data_path, self.model_name))
            self.CH3['t_idx'] = self.CH3['year'].apply(lambda x: str(x)) + self.CH3['month'].apply(
                lambda x: str(x).zfill(2))
            self.CH3 = self.CH3[['t_idx', 'smb', 'vmg', 'mktrf']]

        elif self.model_name == 'CH3_Q.pkl':
            self.CH3 = pd.read_pickle(os.path.join(self.data_path, self.model_name))
            self.CH3['t_idx'] = self.CH3['quarter_code']
            self.CH3 = self.CH3[['t_idx', 'smb', 'vmg', 'mktrf']]

        elif self.model_name == 'FF3_M.pkl':
            self.CH3 = pd.read_pickle(os.path.join(self.data_path, self.model_name))
            self.CH3 = self.CH3[['t_idx', 'smb', 'vmg', 'mktrf']]

        return

    def CH3adjust(self, R: pd.DataFrame, t_index='month_idx', r_index='HML', if_plot=False):
        """
        对输入的序列进行CH3调整，R是一个dataframe，有一列为t_indx时间戳，如果是日度为pd.datatime格式，如果是月度为YYYYMM字符串
        用于和数据进行合并
        :param R:
        :return:
        """
        self.CH3.rename(columns={'t_idx': t_index}, inplace=True)

        R = pd.merge(R, self.CH3, on=[t_index], how='inner') \
            [[t_index, r_index, 'smb', 'vmg', 'mktrf']] \
            .set_index(t_index)

        Y = R[r_index]
        keep_idx = ~np.isnan(Y)

        R = R.loc[keep_idx, :]
        X = sm.add_constant(R[['smb', 'vmg', 'mktrf']])
        Y = R[r_index]

        model = sm.OLS(Y, X)

        # Newy west调整
        result = model.fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        self.model = result
        alpha = result.params[0]
        t = result.tvalues[0]

        if if_plot:
            self.plot_exposure(result, R)

        return alpha, t

    def CAPMadjust(self, R: pd.DataFrame, t_index='month_idx', r_index='HML', if_plot=False):
        """
        对输入的序列进行CAPM调整，R是一个dataframe，有一列为t_indx时间戳，如果是日度为pd.datatime格式，如果是月度为YYYYMM字符串
        用于和数据进行合并
        :param R:
        :return:
        """
        self.CH3.rename(columns={'t_idx': t_index}, inplace=True)

        R = pd.merge(R, self.CH3, on=[t_index], how='inner') \
            [[t_index, r_index, 'mktrf']] \
            .set_index(t_index)

        Y = R[r_index]
        keep_idx = ~np.isnan(Y)

        R = R.loc[keep_idx, :]
        X = sm.add_constant(R[['mktrf']])
        Y = R[r_index]

        model = sm.OLS(Y, X)
        result = model.fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        self.model = result
        alpha = result.params[0]
        t = result.tvalues[0]

        if if_plot:
            self.plot_exposure(result, R)

        return alpha, t

    def plot_exposure(self, reg_result, R):
        """
        对于分解的结果，将exposure和risk factor进行结合，可视化组合收益的来源
        :param R: 时间戳为索引的收益率序列和factor序列
        :param risk_exposure: 回归系数

        :return:
        """
        R = R.reset_index()
        risk_exposure = reg_result.params

        if len(risk_exposure) == 1:
            X = sm.add_constant(R[['mktrf']])
            X.columns = ['alpha', 'mktrf']
        else:
            X = sm.add_constant(R[['smb', 'vmg', 'mktrf']])
            X.columns = ['alpha', 'smb', 'vmg', 'mktrf']
        table = pd.DataFrame({'exposures': [round(x, 3) for x in reg_result.params],
                              't-value': [round(x, 3) for x in reg_result.tvalues]},
                             index=X.columns).T

        X_with_exposure = pd.DataFrame(risk_exposure.values.reshape(1, -1) * X.values, columns=X.columns)
        X_acc = X.cumsum()
        X_with_exposure_acc = X_with_exposure.cumsum()
        R_acc = pd.DataFrame(R.iloc[:, 1]).cumsum()

        # ---- plot
        fig = plt.figure(figsize=(15, 12))

        # - fig1 : return with exposure
        ax1 = fig.add_subplot(2, 1, 1)
        R_acc_syn = np.zeros_like(R_acc.values).reshape(-1)
        for name in X_acc.columns:
            plt.plot(X_with_exposure_acc[name], marker='o')
            R_acc_syn += X_with_exposure_acc[name].values.reshape(-1)
        plt.plot(R_acc, marker='v')
        plt.plot(R_acc_syn, '--vr')
        plt.legend(list(X_with_exposure_acc.columns) + ['portfolio return'] + ['portfolio return syn'])
        plt.title('acc return with exposure')
        plt.ylabel('acc ret')
        plt.xticks([])
        plt.table(cellText=table.values, rowLabels=table.index, colLabels=table.columns)

        # - fig2 : risk factor return
        ax2 = fig.add_subplot(2, 1, 2)
        for name in list(X_acc.columns)[1:]:
            plt.plot(X_acc[name], marker='o')
        plt.legend(list(X_acc.columns)[1:])
        plt.title('acc return of risk factor')
        plt.ylabel('acc ret')
        plt.xticks(list(range(len(X_acc)))[::4], list(R.iloc[:, 0])[::4])

        plt.suptitle(f'Risk Adjust Plot, rsq = {reg_result.rsquared}', size=15)
        plt.show()

        return


def get_mispricing_score(factor_dict, names=None, rank_filter=1, directions=None):
    """
    利用输入的因子进行mispricing分数的加工，可以选择rank时的filter以及加权时各个因子的权重
    :param factor_dict:
    :param rank_filter:
    :param weight:
    :return:
    """
    if names is None:
        names = list(factor_dict['DATA'].keys())

    if directions is None:
        directions = [1 for name in names]
    else:
        if len(directions) != len(names):
            raise NotImplementedError('# of weight != # of factors, please check')
        else:
            pass

    score_mat = np.zeros_like(factor_dict['DATA'][names[0]]).astype(np.float64)
    counting_mat = np.zeros_like(factor_dict['DATA'][names[0]]).astype(np.float64)

    for idx, name in enumerate(names):
        d = directions[idx]
        rank_i = pd.DataFrame(factor_dict['DATA'][name] * rank_filter * d).rank(axis=1, pct=True)
        score_mat += (rank_i.fillna(0.0).values + 1e-7)
        counting_mat += ((~np.isnan(rank_i.values)).astype(np.float64) + 1e-7)

    # 限制条件，只有当因子数量的40%以下为nan时，才包含这只票
    output_mat = np.where(counting_mat > np.ceil(len(names) * 0.4), score_mat / counting_mat, np.nan)
    output_dict = {
        'DATE': factor_dict['DATE'],
        'CODE': factor_dict['CODE'],
        'DATA': {'mispricing': output_mat}
    }

    return output_dict


def get_factor_corr(factor_dict: dict, names: list):
    if len(names) != 2:
        raise ValueError('请输入正确数量的因子名称!')
    factor1 = factor_dict[names[0]]
    factor2 = factor_dict[names[1]]

    factor1_1d = factor1.reshape(-1)
    factor2_1d = factor2.reshape(-1)
    nan_idx = np.isnan(factor1_1d) | np.isnan(factor2_1d)
    factor1_1d_ = factor1_1d[~nan_idx]
    factor2_1d_ = factor2_1d[~nan_idx]

    corr = pd.Series(factor1_1d_).corr(pd.Series(factor2_1d_))
    print(f'expending corr = {corr}')

    corrs = []
    for i in range(factor1.shape[0]):
        data1 = factor1[i, :]
        data2 = factor2[i, :]
        keep = ~(np.isnan(data1) | np.isnan(data2))

        data1 = data1[keep]
        data2 = data2[keep]
        corrs.append(
            pd.Series(data1).corr(pd.Series(data2))
        )
    return corrs


class FamaMacbeth(BaseEstimator):
    __doc__ = """
    Fama Macbeth Regression
    
    Input: 
        1) X array-like, [time by entity] multi-index,
        2) y array-like, [time by entity] multi-index, same lenth as X
    
    Method:
        .fit(y,X) : perform fama-macbeth regression with X,y
        .all_params(): get all params estimation within regression
    Attribute:
        .summary: fama-macbeth regression result summary
    
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, y, X):
        """
        perform fama macbeth regression
        """
        # format check
        if len(X) != len(y):
            raise Exception('Lenth of X and y are not the same, Check Please')
        # try:
        #     assert (X.index.all() == y.index.all())
        # except Exception as e:
        #     print(e)
        #     print('Error with index setting, please keep index of X and index of y exactly the same!')

        # nan cleaning
        X.index.names, y.index.names = ['time', 'entity'], ['time', 'entity']
        X = X.reset_index()
        y = y.reset_index()
        Xy = X.merge(y, on=['time', 'entity'], how='inner').dropna()
        X = Xy[X.columns.to_list()]
        y = Xy[y.columns.to_list()]

        # initialize cross-sectional regression
        dts = list(set(X['time']))
        fm_coef = pd.DataFrame(
            np.ones(
                shape=(len(dts), len(X.columns) - 1)
            ),
            index=dts,
            columns=['constant'] + X.columns.to_list()[2:]
        )
        fm_r2_adj = []
        fm_number = []

        # cross-sectional-regression
        for dt in tqdm(dts):
            X_t = sm.add_constant(X[X['time'] == dt].iloc[:, 2:])
            y_t = y[y['time'] == dt].iloc[:, 2:]
            results = sm.OLS(y_t, X_t).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
            fm_coef.loc[dt, :] = results.params.values
            fm_r2_adj.append(results.rsquared_adj)
            fm_number.append(results.nobs)
        self.__all_params = fm_coef
        self.__all_params['r2_adj'] = fm_r2_adj
        self.__all_params['nobs'] = fm_number
        self.__all_params.sort_index(inplace=True, ascending=True)

        # Newy-West T
        self._summary()

        return self

    def _summary(self):
        """
        perform newy-west t test for FM result
        """
        summary = NewyWestTstats().fit(self.__all_params.iloc[:, :-2], axis=0)
        self.summary = summary
        return

    def all_params(self):
        return self.__all_params


print('--------- Factor Matrix Platform Initiating finished! ----------')
print('------------------------- Version 1.9 --------------------------')
print('------------------------- Author @Lzy --------------------------')
