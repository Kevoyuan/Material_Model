import csv
import os
import numpy as np


#parameterファイルを読み取ってNNC用のYLDパラメータのパスリストを作成する

def P3():
    parameter_success_csv = './parameter_success.csv'                         #パラメータcsvファイル名

    #csv書き込み関数
    def write_csv(csv_name, data_list):
        with open(csv_name,"w",encoding = "shift_jis",newline = "\n") as f:
            writer = csv.writer(f,lineterminator="\n")
            writer.writerows(data_list)

    #カンマ区切りのcsvデータの読み取り
    def read_data(p,header=1) -> np.ndarray:
        #csvデータの読み取り　データ欠損部は0で埋める
        print(os.getcwd())
        a = np.genfromtxt(p, delimiter=',', filling_values=0, encoding='sjis',skip_header=header)
        return a


    #YLD2000-2dのparameterをリストに読み取り

    parameters = read_data(parameter_success_csv,1) 
    key_No = parameters[:,0]
    ad_list= parameters[:,9:18]
    # pprint(key_No)
    # pprint(ad_list)

    # 標準化のための標準偏差standard deviationと平均meanの計算．その後標準化して，dataset_outp.csvに保存
    # n_std,n_meanの初期化
    n_std = np.zeros(9)
    n_mean = np.zeros(9)
    for i in range(9):    # M,alpha1-8の標準偏差と平均を計算し，ndarrayに格納
        n_std[i]=np.std(ad_list[:,i])
        n_mean[i]=np.mean(ad_list[:,i])
    # print(n_std,n_mean)
    n_mag=n_std
    n_offset=n_mean

    os.makedirs('./dataset_outp', exist_ok = True)

    # creation of a csv file for dataset_output
    csv_name = './dataset_outp/dataset_outp.csv'
    j=[(ad_list[k]-n_offset)/n_mag for k in range(len(ad_list))]  #標準化．write_csvのエラー回避のためリスト形式に合わせる

    output = [["y__0:M","y__1:alpha1","y__2:alpha2","y__3:alpha3","y__4:alpha4","y__5:alpha5","y__6:alpha6","y__7:alpha7","y__8:alpha8"]]
    output += j
    write_csv(csv_name, output)

    # save Standardization parameter
    csv_name = './dataset_outp/standardization_parameter.csv'

    output2 = [["","M","alpha1","alpha2","alpha3","alpha4","alpha5","alpha6","alpha7","alpha8"]]
    output2 += [["std"]+n_std.tolist()]     # リスト形式
    output2 += [["mean"]+n_mean.tolist()]    
    write_csv(csv_name, output2)


#実行部
if __name__ == '__main__':
    P3()        #読み取るパラメータcsvファイル名をに入力

