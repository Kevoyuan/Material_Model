import numpy as np
from pathlib import Path
from pprint import pprint
import csv
import sympy as sp
import time
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

#csv書き込み関数
def write_csv(csv_name, data_list):
    with open(csv_name,"w",encoding = "shift_jis",newline = "\n") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerows(data_list)

#カンマ区切りのcsvデータの読み取り
def read_data(p: Path,header=1) -> np.ndarray:
    #csvデータの読み取り　データ欠損部は0で埋める
    a = np.genfromtxt(p, delimiter=',', filling_values=0, encoding='sjis',skip_header=header)
    return a

#parameter_success.csvから番号を読み取り
def read_parameter(p :Path) -> list:
    #YLD2000-2dの計算結果や入力した実験値をリストに読み取り
    with open(p,'r') as paraf :
        para_read = csv.reader(paraf)
        parameter_list = [a for a in para_read]
        output_No = [i[0] for i in parameter_list]      #Noの読み取り
        para = [i[9::] for i in parameter_list]         #Yld2000-2dパラメータの読み取り
        output_No.pop(0)                                #タイトル行の削除
        para.pop(0)
        #パラメータをstrからfloatに変換
        output_para = []
        for i in para:
            ad = []
            for j in i:
                ad.append(float(j))
            ad_ad = np.array(ad)
            output_para.append(ad_ad)      
    return output_No, output_para

#百分率誤差の計算
def cal_error_of_percent(correct :np.array, predict :np.array) -> list:
    error = (np.abs(predict - correct) / correct*100)   #誤差計算
    e_max = np.max(np.delete(error, 0))                 #Mを除いた誤差の最大値
    e_min = np.min(np.delete(error, 0))                 #Mを除いた誤差の最小値
    e_ave = np.average(np.delete(error, 0))             #Mを除いた誤差の平均値
    return [e_max, e_min, e_ave]

#Yld2000-2dのX応力,Y応力の取得
def cal_sig_x_y(ratio, M, a1, a2, a3, a4, a5, a6, a7, a8):
    if ratio == 0:
        ratio = 1e-7
    # sig_x = (3*(2**(1/M))) / (((2*a1 + a2) + (-a1 -2*a2)*ratio)**M + ((2*a3 - 2*a4) + (-a3 + 4*a4)*ratio)**M + ((4*a5 -a6) + (-2*a5 + 2*a6)*ratio)**M)**(1/M)
    # sig_y = (3*(2**(1/M))) / (((2*a1 + a2)*(1/ratio) + (-a1 -2*a2))**M + ((2*a3 - 2*a4)*(1/ratio) + (-a3 + 4*a4))**M + ((4*a5 -a6)*(1/ratio) + (-2*a5 + 2*a6))**M)**(1/M)
    sig_x = (3*(2**(1/M))) / (abs((2*a1 + a2) + (-a1 -2*a2)*ratio)**M + abs((2*a3 - 2*a4) + (-a3 + 4*a4)*ratio)**M + abs((4*a5 -a6) + (-2*a5 + 2*a6)*ratio)**M)**(1/M)
    sig_y = ratio * sig_x

    return [sig_x, sig_y]

#Yld2000-2dの等二軸応力,XY応力の取得
def cal_sig_b_xy(ratio, M, a1, a2, a3, a4, a5, a6, a7, a8):
    if ratio == 0:
        ratio = 1e-7
    # sig_b = (2/ ((1/9*(a1 - a2)**2 + (2*a7*ratio)**2)**(M/2) +(1/6)**M*( ((a3 + 2*a4 + 2*a5 + a6) - ((a3 + 2*a4 - 2*a5 - a6)**2 + ( 6*a8 *ratio)**2)**0.5)**M + ((a3 + 2*a4 + 2*a5 + a6) + ((a3 + 2*a4 - 2*a5 - a6)**2 + ( 6*a8 *ratio)**2)**0.5)**M)))**(1/M)
    sig_b = (2/ (abs(1/9*(a1 - a2)**2 + (2*a7*ratio)**2)**(M/2) +(1/6)**M*( abs((a3 + 2*a4 + 2*a5 + a6) - ((a3 + 2*a4 - 2*a5 - a6)**2 + ( 6*a8 *ratio)**2)**0.5)**M + abs((a3 + 2*a4 + 2*a5 + a6) + ((a3 + 2*a4 - 2*a5 - a6)**2 + ( 6*a8 *ratio)**2)**0.5)**M)))**(1/M)
    sig_xy = ratio * sig_b
    return [sig_b, sig_xy]

#X-Y応力点の距離の誤差計算
def cal_error_of_radius(correct :np.array, predict :np.array) -> list:

    a_se = np.empty((0,2), dtype=object)
    for i in range(0, 95, 5):                                                       #シータを0degから90degまで5deg間隔
        #シータから応力比の取得
        ratio = np.tan(np.radians(i))
        if ratio == 0:
            ratio = 1e-7
        
        #比較するYld2000-2dパラメータ
        parameter1 = [round(correct[0]), correct[1], correct[2], correct[3], correct[4], correct[5], correct[6], correct[7], correct[8]]
        parameter2 = [round(predict[0]), predict[1], predict[2], predict[3], predict[4], predict[5], predict[6], predict[7], predict[8]]
        parameter1.insert(0, ratio)                                                 #Yldパラメータに応力比の追加
        parameter2.insert(0, ratio)

        #Yld2000-2dの応力点の取得 Yld2000-2dパラメータと応力比を入力
        #X応力,Y応力
        sig1_x_y = cal_sig_x_y(*parameter1)
        sig2_x_y = cal_sig_x_y(*parameter2)
        #等二軸応力,XY応力
        sig1_b_xy = cal_sig_b_xy(*parameter1)
        sig2_b_xy = cal_sig_b_xy(*parameter2)
        #Yld2000-2dの半径rの取得
        r1_x_y = (sig1_x_y[0]**2 + sig1_x_y[1]**2) ** (1 / 2)                       #極座標の半径rを取得
        r2_x_y = (sig2_x_y[0]**2 + sig2_x_y[1]**2) ** (1 / 2)
        r1_b_xy = (sig1_b_xy[0]**2 + sig1_b_xy[1]**2) ** (1 / 2)
        r2_b_xy = (sig2_b_xy[0]**2 + sig2_b_xy[1]**2) ** (1 / 2)
        se = np.array([(r2_x_y - r1_x_y)**2, (r2_b_xy - r1_b_xy)**2])               #二乗誤差
        #a_se = np.append(a_se, se)
        a_se=np.vstack([a_se, se])
    rmse = (np.mean(a_se,axis=0)) ** (1 / 2)                                               #RMSE

    #各種誤差の結合
    error = [rmse[0],rmse[1]]

    return error

#誤差計算の並列実行用関数
def cal_error_exe(arg_list):
    error_of_percent = cal_error_of_percent(arg_list[0], arg_list[1])               #Mを除いたα誤差[誤差の最大値, 誤差の最小値, 誤差の平均値]
    error_of_distance = cal_error_of_radius(arg_list[0], arg_list[1])               #降伏曲面の半径誤差[RMSE]
    return [arg_list[2]] + error_of_percent + error_of_distance 

#誤差のcsvファイルを回収し結合
def collect_error_csv(nn_list :list, output_path):
    #各error.csvの読み取り
    df_list = []   
    for i in nn_list:
        p = Path(i, 'error.csv')
        a = pd.read_csv(p, header=None)
        df_list.append(a)
    #error.csv同士の結合
    output = df_list[0]
    for i in range(len(df_list) - 1):
        output = pd.concat([output, df_list[i + 1]], axis=1)    #横方向に結合
    #csv書出し
    output.to_csv(output_path, encoding='sjis', header=False, index=False)    #pandas_dataflameの列・行名は出力しない
    return

#実行関数
def main():
    #誤差計算を行うNNCのパス
    nn_list = [Path(r"D:\LS-DYNA_data\nnc\test004\NN_works4_CNN\nnc_files\hourglass_test004_406_012_6_2D_x_y_strain.files\20221108_143543")]
            #    Path(r"D:\LS-DYNA_data\nnc\test004\NN_works\nnc_files\hourglass_test004_102_002_3input_xstrain.files\20220913_123438")]
    # nn_list = [Path(r"D:\LS-DYNA_data\nnc\test002\NN_works\nnc_files\hourglass_test002_2layer_06.files\20220725_131136")]
             
    pbar = tqdm(total=len(nn_list))                                         #プログレスバーの定義
    all_csv = []
    for l in nn_list:
        #予測結果'output_result.csv'から，正解および予測のパラメータを取り出す
        nnc_result_path = Path(l, 'output_result.csv')
        parameter_in_resultfile = read_data(nnc_result_path,1) 
        parameter_No=np.arange(len(parameter_in_resultfile))
        n_mag=np.array([10,2,2,2,2,2,2,2,2])
        n_offset=np.array([0,0,0,0,0,0,0,0,0])
        correct_parameter= [i*n_mag+n_offset for i in parameter_in_resultfile[:,9:18]]              #予測値にn_magを掛け，n_offsetを足して戻す
        predict_parameter= [i*n_mag+n_offset for i in parameter_in_resultfile[:,18:27]]              #予測値にn_magを掛け，n_offsetを足して戻す

        #csvファイルのタイトル行
        output = [[Path(l.parts[-2], l.parts[-1]), 'percentage_error(without_M)','','','radial_distance_error',''], 
                ['n', 'Max', 'Min', 'Ave', 'RMSE1','RMSE2']]
        #正解パラメータ,予測パラメータ,パラメータNoの結合
        arg_list = [[i, j, k] for i, j, k in zip(correct_parameter, predict_parameter, parameter_No)]

        #誤差計算の実行
        cpus = 10
        with Pool(cpus) as p :
            result = p.map(cal_error_exe,arg_list)
        output += result
        write_csv(Path(l, 'error.csv'), output) #csv書出し
        pbar.update(1)                          #プログレスバーの更新
    pbar.close()
    
    #誤差のcsvファイルを回収・結合・書出し
    collect_error_csv(nn_list, Path(r'./all_errors.csv'))

#以下実行部
if __name__ == '__main__':
    main()
