#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#カンマ区切りのcsvデータの読み取り
def read_data(p: Path,header=1) -> np.ndarray:
    #csvデータの読み取り　データ欠損部は0で埋める
    a = np.genfromtxt(p, delimiter=',', filling_values=0, encoding='sjis',skip_header=header)
    return a

def read_parameter(nnc_result_path, parameter_No):

    #予測結果'output_result.csv'から，正解および予測のパラメータを取り出す
    parameter_in_resultfile = read_data(nnc_result_path,1) 
    # parameter_No=np.arange(len(parameter_in_resultfile))
    n_mag=np.array([10,2,2,2,2,2,2,2,2])
    n_offset=np.array([0,0,0,0,0,0,0,0,0])
    correct_parameter= [i*n_mag+n_offset for i in parameter_in_resultfile[:,9:18]]              #予測値にn_magを掛け，n_offsetを足して戻す
    predict_parameter= [i*n_mag+n_offset for i in parameter_in_resultfile[:,18:27]]              #予測値にn_magを掛け，n_offsetを足して戻す
    
    M_01 = correct_parameter[parameter_No][0]   
    alpha_01 = correct_parameter[parameter_No][1:]
    M_02 = predict_parameter[parameter_No][0]   
    alpha_02 = predict_parameter[parameter_No][1:]

    return M_01, alpha_01, M_02, alpha_02

#Yld2000-2dのX応力,Y応力の取得
def cal_sig_x_y(ratio, M, a1, a2, a3, a4, a5, a6, a7, a8):
    sig_x = (3*(2**(1/M))) / (abs((2*a1 + a2) + (-a1 -2*a2)*ratio)**M + abs((2*a3 - 2*a4) + (-a3 + 4*a4)*ratio)**M + abs((4*a5 -a6) + (-2*a5 + 2*a6)*ratio)**M)**(1/M)
    sig_y = ratio * sig_x
    return [sig_x.flatten().tolist()+sig_y.flatten().tolist()]

#Yld2000-2dの等二軸応力,XYせん断応力の取得
def cal_sig_b_xy(ratio, M, a1, a2, a3, a4, a5, a6, a7, a8):
    sig_b = (2/ (abs(1/9*(a1 - a2)**2 + (2*a7*ratio)**2)**(M/2) +(1/6)**M*( abs((a3 + 2*a4 + 2*a5 + a6) - ((a3 + 2*a4 - 2*a5 - a6)**2 + ( 6*a8 *ratio)**2)**0.5)**M + abs((a3 + 2*a4 + 2*a5 + a6) + ((a3 + 2*a4 - 2*a5 - a6)**2 + ( 6*a8 *ratio)**2)**0.5)**M)))**(1/M)
    sig_xy = ratio * sig_b
    return [sig_b.flatten().tolist() + sig_xy.flatten().tolist()]


parameter_No=38
nnc_result_path =Path(r"D:\LS-DYNA_data\nnc\test004\NN_works4_CNN\nnc_files\hourglass_test004_406_012_6_2D_x_y_strain.files\20221108_143543\output_result.csv")

file_end="_004_406_12_6.png"

# nnc_result_pathから，parameterのNoを指定して，正解と予測値のM，αを取得
M_01, alpha_01, M_02, alpha_02 = read_parameter(nnc_result_path, parameter_No)


# 読み込んだ値の確認
print(M_01)
print(M_02)


# 読み込んだ値の確認
print(alpha_01)
print(alpha_02)

# 自分の値で描画したいときは，ここのセルを有効にして上書き
# M_01=6
# alpha_01=[1.080,1.041,1.066,0.9449,0.9558,1.130,0.9913,0.7802]

# M_02=6
# alpha_02=[1.080,1.041,1.066,0.9449,0.9558,1.130,0.9913,0.7802]

sigma1=[]
sigma2=[]
sigma1_bxy=[]
sigma2_bxy=[]

for i in range(0, 91, 1):                                                       #シータを0degから90degまで1deg間隔
        #シータから応力比の取得
        ratio = np.tan(np.radians(i))
        
        #比較するYld2000-2dパラメータ
#        parameter1 = [ratio, round(M_01), alpha_01[0], alpha_01[1], alpha_01[2], alpha_01[3], alpha_01[4], alpha_01[5], alpha_01[6], alpha_01[7]]
        parameter1 = [ratio, round(M_01)]+[alpha_01[i] for i in range(8)]
#         parameter2 = [ratio, round(M_02)]+[alpha_02[i] for i in range(8)]
        parameter2 = [ratio, M_02]+[alpha_02[i] for i in range(8)]

        #Yld2000-2dの応力点の取得 Yld2000-2dパラメータと応力比を入力
        #X応力,Y応力

        sig1_x_y = cal_sig_x_y(*parameter1)
        sig2_x_y = cal_sig_x_y(*parameter2)

        sigma1.extend(sig1_x_y)
        sigma2.extend(sig2_x_y)


        #Yld2000-2dの応力点の取得 Yld2000-2dパラメータと応力比を入力
        #等二軸応力,XYせん断応力
        sig1_b_xy = cal_sig_b_xy(*parameter1)
        sig2_b_xy = cal_sig_b_xy(*parameter2)

        sigma1_bxy.extend(sig1_b_xy)
        sigma2_bxy.extend(sig2_b_xy)


sig1_x1 = np.array(sigma1)[:,0]
sig1_y1 = np.array(sigma1)[:,1]
sig2_x1 = np.array(sigma2)[:,0]
sig2_y1 = np.array(sigma2)[:,1]
#
sig1_x2 = np.array(sigma1_bxy)[:,0]
sig1_y2 = np.array(sigma1_bxy)[:,1]
sig2_x2 = np.array(sigma2_bxy)[:,0]
sig2_y2 = np.array(sigma2_bxy)[:,1]
    

# figureを生成する
fig = plt.figure(figsize=(12, 6))

plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix'       # math fontの設定
plt.rcParams["font.size"] = 15                  # 全体のフォントサイズ
plt.rcParams['xtick.labelsize'] = 14            # x軸のフォントサイズ
plt.rcParams['ytick.labelsize'] = 14            # y軸のフォントサイズ

# axをfigureに設定する
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
 
# ax1の定義
ax1.plot(sig1_x1,sig1_y1, label="Answer") # x1, y1をプロット
ax1.plot(sig2_x1,sig2_y1, label="NN prediction") # x1, y1をプロット

ax1.set_xlabel(r'$\sigma_x$')
ax1.set_ylabel(r'$\sigma_y$')

# ax2の定義
ax2.plot(sig1_x2,sig1_y2, label="True") # x1, y1をプロット
ax2.plot(sig2_x2,sig2_y2, label="NN") # x2, y2をプロット

ax2.set_xlabel(r'$\sigma_b$')
ax2.set_ylabel(r'$\sigma_{xy}$')

ax2.set_title("Parameter No.="+str(parameter_No))
plt.legend()

# figureの保存
# fig.savefig("file_No_"+str(file_No)+file_end,dpi=130,facecolor="white", edgecolor="coral")

plt.show()


