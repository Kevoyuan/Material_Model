import numpy as np
from pathlib import Path
import csv
import os

#csv書き込み関数
def write_csv(csv_name, data_list):
    with open(csv_name,"w",encoding = "shift_jis",newline = "\n") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerows(data_list)

#カンマ区切りのcsvデータの読み取り
def read_data(p: Path) -> np.ndarray:
    #csvデータの読み取り　データ欠損部は0で埋める
    #a = np.genfromtxt(p, delimiter=',', filling_values=0, encoding='sjis')
    # a = np.genfromtxt(p, delimiter=',', filling_values=0, encoding='utf8')
    a = np.genfromtxt(p, delimiter=',', encoding='utf_8_sig')  
    return a

#2列の実験データの加工
def edit_ex_data_2row(p1 :Path, p2 :Path) -> np.ndarray:
    #実験データ
    a_ex = read_data(p1)    #[Xdata, Ydata]
    #基準データ
    a_ref = read_data(p2)

    Ydata = np.interp(a_ref, a_ex[:, 0], a_ex[:, 1])      #補間されたYdata


    #各データ配列をn行1列のデータに変換
    Ydata_T = Ydata.reshape(len(Ydata), 1)
    a_ref_T = a_ref.reshape(len(a_ref), 1)

    #列の結合
    output = np.hstack([a_ref_T, Ydata_T])
    return output

#3列の実験データの加工
def edit_ex_data_3row(p1 :Path, p2 :Path) -> np.ndarray:
    #実験データ
    a_ex = read_data(p1)    #[X変位, Y変位, Y荷重]
    #基準データ
    a_ref = read_data(p2)

    Xdisp = np.interp(a_ref, a_ex[:, 1], a_ex[:, 0])      #補間されたX変位
    Yforce = np.interp(a_ref, a_ex[:, 1], a_ex[:, 2])     #補間されたY荷重

    #各データ配列をn行1列のデータに変換
    Xdisp_T = Xdisp.reshape(len(Xdisp), 1)
    Yforce_T = Yforce.reshape(len(Yforce), 1)
    a_ref_T = a_ref.reshape(len(a_ref), 1)

    #列の結合
    output = np.hstack([Xdisp_T, a_ref_T, Yforce_T])
    return output

#4列の実験データの加工
def edit_ex_data_4row(p1 :Path, p2 :Path) -> np.ndarray:
    #実験データ
    a_ex = read_data(p1)    #[X変位1, X変位2, Y変位, Y荷重]
    #基準データ
    a_ref = read_data(p2)

    Xdisp1 = np.interp(a_ref, a_ex[:, 2], a_ex[:, 0])     #補間されたX変位1
    Xdisp2 = np.interp(a_ref, a_ex[:, 2], a_ex[:, 1])     #補間されたX変位2
    Yforce = np.interp(a_ref, a_ex[:, 2], a_ex[:, 3])     #補間されたY荷重

    #各データ配列をn行1列のデータに変換
    Xdisp1_T = Xdisp1.reshape(len(Xdisp1), 1)
    Xdisp2_T = Xdisp2.reshape(len(Xdisp2), 1)
    Yforce_T = Yforce.reshape(len(Yforce), 1)
    a_ref_T = a_ref.reshape(len(a_ref), 1)

    #列の結合
    output = np.hstack([Xdisp1_T, Xdisp2_T, a_ref_T, Yforce_T])
    return output

def main():
    #実験データパス
    p_ex_path = Path(r"D:\LS-DYNA_data\nnc\test004\experiment\220620_HW_thickness_reduction_csv")
    #基準データパス
    p_ref = Path(r"D:\LS-DYNA_data\nnc\test004\experiment\inference_data\reference_strain.csv")
    #リサンプリングデータパス
    resamp_path="resampling"

    os.chdir(p_ex_path)
    os.makedirs(resamp_path, exist_ok = True)
    os.chdir(resamp_path)

    for item in p_ex_path.glob('./*.csv'):
        print(item)
        output = edit_ex_data_2row(item, p_ref)         #2列のデータ用

        #output = edit_ex_data_3row(item, p_ref)        #3列のデータ用
        #output = edit_ex_data_4row(item, p_ref)        #4列のデータ用
        
        np.savetxt('resampling_'+item.name, output, delimiter=',')
    return

#以下実行部
if __name__ == '__main__':
    main()