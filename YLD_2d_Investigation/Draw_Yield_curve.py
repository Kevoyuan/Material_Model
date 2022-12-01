#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_parameter(file_name, header=1):
    # YLD2000-2dの正解値やNNの予測結果をリストに読み取り
    para = np.genfromtxt(file_name, delimiter=",", skip_header=header)[
        8:
    ]  # 0-7:orijinal experimental data

    # 正規化されたパラメータをもとに戻す Restore standardized values.
    n_mag = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    n_offset = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    para = para * n_mag + n_offset

    M = para[0]
    alpha = para[1:]

    return M, alpha


# Yld2000-2dのX応力,Y応力の取得
def cal_sig_x_y(ratio, M, a1, a2, a3, a4, a5, a6, a7, a8):
    sig_x = (3 * (2 ** (1 / M))) / (
        abs((2 * a1 + a2) + (-a1 - 2 * a2) * ratio) ** M
        + abs((2 * a3 - 2 * a4) + (-a3 + 4 * a4) * ratio) ** M
        + abs((4 * a5 - a6) + (-2 * a5 + 2 * a6) * ratio) ** M
    ) ** (1 / M)
    sig_y = ratio * sig_x
    return [sig_x.flatten().tolist() + sig_y.flatten().tolist()]


# Yld2000-2dの等二軸応力,XYせん断応力の取得
def cal_sig_b_xy(ratio, M, a1, a2, a3, a4, a5, a6, a7, a8):
    sig_b = (
        2
        / (
            abs(1 / 9 * (a1 - a2) ** 2 + (2 * a7 * ratio) ** 2) ** (M / 2)
            + (1 / 6) ** M
            * (
                abs(
                    (a3 + 2 * a4 + 2 * a5 + a6)
                    - ((a3 + 2 * a4 - 2 * a5 - a6) ** 2 + (6 * a8 * ratio) ** 2) ** 0.5
                )
                ** M
                + abs(
                    (a3 + 2 * a4 + 2 * a5 + a6)
                    + ((a3 + 2 * a4 - 2 * a5 - a6) ** 2 + (6 * a8 * ratio) ** 2) ** 0.5
                )
                ** M
            )
        )
    ) ** (1 / M)
    sig_xy = ratio * sig_b
    return [sig_b.flatten().tolist() + sig_xy.flatten().tolist()]


# Change the number of file


def export_yield_curve(path, name):
    # true file name and read parameters
    file_01 = f"{path}/{name}.csv"
    M_01, alpha_01 = read_parameter(file_01)

    # 読み込んだ値の確認
    print(M_01)
    print(alpha_01)

    # If you want to draw with your own values, enable following comments and overwrite the values here.
    # M_01=8
    # alpha_01=[1.080,1.041,1.066,0.9449,0.9558,1.130,0.9913,0.7802]

    # M_02=8
    # alpha_02=[1.080,1.041,1.066,0.9449,0.9558,1.130,0.9913,0.7802]

    sigma1 = []

    sigma1_bxy = []

    for i in range(0, 91, 1):  # シータを0degから90degまで1deg間隔
        # シータから応力比の取得
        ratio = np.tan(np.radians(i))

        # 比較するYld2000-2dパラメータ
        #        parameter1 = [ratio, round(M_01), alpha_01[0], alpha_01[1], alpha_01[2], alpha_01[3], alpha_01[4], alpha_01[5], alpha_01[6], alpha_01[7]]
        parameter1 = [ratio, M_01] + [alpha_01[i] for i in range(8)]

        # Yld2000-2dの応力点の取得 Yld2000-2dパラメータと応力比を入力
        # X応力,Y応力

        sig1_x_y = cal_sig_x_y(*parameter1)

        sigma1.extend(sig1_x_y)

        # Yld2000-2dの応力点の取得 Yld2000-2dパラメータと応力比を入力
        # 等二軸応力,XYせん断応力
        sig1_b_xy = cal_sig_b_xy(*parameter1)

        sigma1_bxy.extend(sig1_b_xy)

    sig1_x1 = np.array(sigma1)[:, 0]
    sig1_y1 = np.array(sigma1)[:, 1]

    #
    sig1_x2 = np.array(sigma1_bxy)[:, 0]
    sig1_y2 = np.array(sigma1_bxy)[:, 1]

    df = pd.DataFrame(
        {
            "sig_x/sig_0": sig1_x1,
            "sig_y/sig_0": sig1_y1,
            "sig_b/sig_0": sig1_x2,
            "sig_xy/sig_0": sig1_y2,
        }
    )

    df.to_csv(f"{path}/{name}_yield.csv")
