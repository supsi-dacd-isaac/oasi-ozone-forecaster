import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


class ComparisonUtils:

    @staticmethod
    def handle_quantiles(data, meas, region, predictor, quantiles):
        df_list = []
        for quantile in quantiles:
            df_tmp = data[(meas, (('location', region), ('predictor', predictor), ('quantile', quantile)))]
            df_tmp.columns = [quantile]
            df_list.append(df_tmp)
        return pd.concat(df_list, axis=1)

    @staticmethod
    def quantile_scores(q_hat: np.ndarray, y: np.ndarray, alphas):
        s_k_alpha, reliability_alpha = [[], []]
        for a, alpha in enumerate(alphas):
            # err = q_hat[:, a, :] - y
            err = q_hat[:, a] - y

            I = np.asanyarray(err > 0, dtype=int)
            s_k = (I - alpha) * err
            s_k_alpha.append(np.mean(s_k, axis=0))
            reliability_alpha.append(np.mean(I, axis=0))

        return {
            'reliability': reliability_alpha,
            'skill': s_k_alpha,
            'qs_50': s_k_alpha[int(len(alphas) / 2) + 1],
            'mae_rel': mean_absolute_error(alphas, reliability_alpha),
            'mae_rel_0-50': mean_absolute_error(alphas[0:int(len(alphas)/2)], reliability_alpha[0:int(len(alphas)/2)]),
            'mae_rel_50-100': mean_absolute_error(alphas[int(len(alphas)/2):], reliability_alpha[int(len(alphas)/2):]),
        }