from influxdb import DataFrameClient
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def format_one_levels_response(response):
    index = []
    for k, v in response.items():
        try:
            index.append(k[1][0][1])
        except:
            pass

    df = {}
    for k, v in zip(index, response.values()):
        try:
            df[k] = v
        except:
            pass
    return pd.concat(df, axis=1)

cfg = {"host": "isaac-db01.dacd.supsi.ch",
    "port": 443,
    "username": "oasi_ro",
    "password": "t49R3I3tSjoJoYQyEki3TgnK",
    "database": "oasi",
    "ssl": True}


def download_data(t_start="2023-01-09T12:00:00Z", t_end="2023-03-09T13:00:00Z"):
    q = "select * from inputs_measurements " \
        "where time>='{}' and time<='{}' " \
        "and signal='PM10' group by location".format(t_start, t_end)

    influx_client = DataFrameClient(**cfg)
    data = format_one_levels_response(influx_client.query(q))
    return data.loc[:, data.columns.get_level_values(1)=='value'].droplevel(1, 1)


def get_hankel(df, embedding=3):
  return pd.concat([df.shift(-l) for l in range(embedding)], axis=1).iloc[:-embedding]


def get_svd_rec(df, embedding, n_pc):
    e = get_hankel(df, embedding)
    U, S, V = np.linalg.svd(e.values, full_matrices=False)
    approx = (U[:, :n_pc] @ np.diag(S[:n_pc]) @ V[:n_pc, :])
    rec = np.hstack([approx[0, :int((embedding-1)/2)], approx[:, int((embedding-1)/2)], approx[-1, int((embedding-1)/2):]])
    return pd.DataFrame(rec, index=df.index, columns=df.columns), approx, e


lagged_mav = lambda x, k: x.copy().rolling('{}d'.format(k)).mean()


def svd_outlier_detector(df, embedding=5, n_sigma=3, detr_days=2, do_plots=True):
    trend = lagged_mav(df.copy(), detr_days)
    df_detr = df.copy()-trend
    df_corrected = df_detr.copy()

    for i in [1, 2, 10]:
        df_rec, approx, e = get_svd_rec(df_corrected, embedding, i)
        df_corrected = df_detr.copy()
        rec_err = np.abs(df_detr-df_rec)
        candidates = rec_err.index[(rec_err > rec_err.std()*n_sigma).values.ravel()]
        rec_err_tot = pd.DataFrame(np.abs(approx-get_hankel(df_detr, embedding)).mean(axis=1).ravel(), index=df.index[int((embedding-1)/2)+1:-int((embedding-1)/2)])
        #candidates = rec_err_tot.index[(rec_err_tot > rec_err_tot.std() * n_sigma).values.ravel()]
        df_corrected.loc[candidates, :] = df_rec.loc[candidates, :]

    df_corrected += trend
    if do_plots:
        fig, ax = plt.subplots(1, 1)
        df.plot(ax=ax)
        df_corrected.plot(ax=ax)
        rec_err_tot.plot(ax=ax)
        rec_err.plot(ax=ax)
        ax.scatter(candidates, df.loc[candidates])
    return df_corrected, candidates


if __name__ == '__main__':
    df = download_data(t_start="2023-01-09T12:00:00Z", t_end="2023-03-09T13:00:00Z")
    df_imputed= df.copy()
    outliers = {}
    for c in df.columns:
        imputed, outs = svd_outlier_detector(df[[c]].interpolate(), embedding=11, n_sigma=7, detr_days=7)
        df_imputed.loc[:, c] = imputed[c]
        outliers[c] = outs

    fig, ax = plt.subplots(1, 1)
    l = df.plot(ax=ax)
    colors = [l._color for l in l.axes.get_lines()]
    [df_imputed.loc[:, column].plot(ax=ax, linestyle='--', color=c) for column, c in zip(df.columns, colors)]
