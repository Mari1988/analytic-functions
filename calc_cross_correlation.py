import os
import pandas as pd
import numpy as np
from itertools import combinations
import pdb

path = '/home/mariappan/Mari/tmp/data/'
os.chdir(path)

def crosscorr(x, y, lag=0):
    '''

    :param x: variable of type pandas-core-series
    :param y: ariable of type pandas-core-series
    :param lag: time periods by which series to be shifted
    :return: correlation between x & y and y & x for the given lag (type: tuple)
    '''
    return x.corr(y.shift(lag)), y.corr(x.shift(lag))

def calc_cross_cor_mat(pdf, lag_max=1):
    '''

    :param pdf: input pandas dataframe for which cross correlation needs to be calculated
    :param lag_max: maximum lag (calculated as 10*log10(n-samples/n-features))
    :return: returns cross correlation matrix (as dataframe) averaged across all lags
    '''
    col_pairs = list(combinations(pdf.columns, 2))
    cor_mat_dict = dict()
    for lag in range(lag_max):

        cor_mat_temp = pd.DataFrame(data=None, index=pdf.columns, columns=pdf.columns)
        for col_pairs_i in col_pairs:
            pdf_s = pdf[list(col_pairs_i)]
            xy_cor, yx_cor = crosscorr(pdf_s[col_pairs_i[0]], pdf_s[col_pairs_i[1]], lag=lag)
            avg_cross_cor = (xy_cor + yx_cor) / 2
            cor_mat_temp.loc[col_pairs_i[0], col_pairs_i[1]] = cor_mat_temp.loc[
                col_pairs_i[1], col_pairs_i[0]] = avg_cross_cor
        np.fill_diagonal(cor_mat_temp.values, 1)
        cor_mat_dict[lag] = cor_mat_temp

    # find average correlation across all lags
    avg_cor_mat = pd.DataFrame(data=0, index=pdf.columns, columns=pdf.columns)
    for key, value in cor_mat_dict.items():
        avg_cor_mat = avg_cor_mat + value
    avg_cor_mat = avg_cor_mat / len(cor_mat_dict)

    return  avg_cor_mat

# read df
pdf = pd.read_csv('./dat_i.csv')
lag_max = int(np.floor(10 * np.log10(pdf.shape[0]/pdf.shape[1])))
avg_cor_mat = calc_cross_cor_mat(pdf, lag_max)
