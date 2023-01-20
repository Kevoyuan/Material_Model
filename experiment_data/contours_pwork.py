import csv
import pathlib
import re
from pprint import pprint

import numpy as np


#検索する相当塑性ひずみの入力
def searched_eq_strain() -> list:
    values = np.array([])
    while True:
        n = float(input('検索する相当塑性ひずみを入力 (0で入力終了) -> '))
        if n == 0:
            break
        elif n != 0:
            values = np.append(values, n)
    print(values)
    return values

#csv書き込み関数
def write_csv(csv_name, data_list):
    with open(csv_name,"w",encoding = "shift_jis",newline = "\n") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerows(data_list)

#csvのpath一覧の取得
def get_csv_paths(path: str) -> pathlib.WindowsPath:
    p = pathlib.Path(path)
    p_all = list(p.iterdir())                                                           #すべてのパス
    p_0 = [i for i in p.glob('00deg.csv')]                                              #0度方向の単軸引張実験データのパスの取得
    p_other = [i for i in p.glob('**/*') if re.search('^(?!.*00deg.csv).*$', str(i))]   #0度方向以外のパスの取得
    print('-----0deg_path------')
    pprint(p_0)
    print('----other_paths----')
    pprint(p_other)
    return p_0, p_other

#カンマ区切りのcsvデータの読み取り
def read_data(p: pathlib.WindowsPath) -> np.ndarray:
    #csvデータの読み取り　データ欠損部は0で埋める
    a = np.genfromtxt(p, delimiter=',', filling_values=0, encoding='sjis')
    #pprint(a)
    return a

#累積塑性仕事の計算
def plastic_work(a: np.ndarray):
    p_work = np.array([0])
    #print(a)
    for i in range(len(a)-1):
        x_work = (a[i + 1, 2] + a[i, 2])*(a[i + 1, 0] - a[i, 0]) / 2
        y_work = np.float64((a[i + 1, 3] + a[i, 3])*(a[i + 1, 1] - a[i, 1]) / 2)
        xy_work = x_work + y_work
        p_work = np.append(p_work, xy_work + p_work[i])
    #print(p_work)
    return p_work

#塑性ひずみ増分方向の計算
def strain_increment(a1: np.ndarray, a2: np.ndarray, num1: float, num2: float) -> float:  #(0deg, other, 検索値, 差分の大きさ)
    #塑性仕事の計算
    a1_pwork = plastic_work(a1)
    a2_pwork = plastic_work(a2)

    a1_p = np.interp([num1 - num2, num1 + num2],  a1[:, 0], a1_pwork)    #a1の塑性仕事を検索[plus, minus]
    a2_strainX = np.interp(a1_p, a2_pwork, a2[:, 0])                              #a2のX塑性ひずみ
    a2_strainY = np.interp(a1_p, a2_pwork, a2[:, 1])                              #a2のY塑性ひずみ
    #差分の分母が0になった場合の回避
    if a2_strainX[1] == a2_strainX[0]:
        increment = 'error'
    else:
        increment = (a2_strainY[1] - a2_strainY[0]) / (a2_strainX[1] - a2_strainX[0])       #塑性ひずみ増分方向
    return increment

#0degの塑性ひずみ増分方向の計算
def strain_increment_00deg(a1: np.ndarray, num1: float, num2: float) -> float:  #(0deg, 検索値, 差分の大きさ)
    a1_epsY = np.interp([num1 - num2, num1 + num2],  a1[:, 0], a1[:, 1])    #a1のyひずみ
    #分子が0になった場合の回避
    if a1_epsY[1] == a1_epsY[0]:
        increment = 'error'
    else:
        increment = (a1_epsY[1] - a1_epsY[0]) / ((num1 + num2) - (num1 - num2)) #ひずみ増分方向
    return increment

#等塑性仕事面の値の検索
def search_values(a1: np.ndarray, a2: np.ndarray, num: float) -> np.ndarray:   #(0deg, other, 検索値)
    #塑性仕事の計算
    a1_pwork = plastic_work(a1)
    a2_pwork = plastic_work(a2)

    a1_p = np.interp([num],  a1[:, 0], a1_pwork)    #a1の塑性仕事を検索
    a2_epsX = np.interp(a1_p, a2_pwork, a2[:, 0])   #a2のx应变
    a2_epsY = np.interp(a1_p, a2_pwork, a2[:, 1])   #a2のy应变
    a2_sgmX = np.interp(a1_p, a2_pwork, a2[:, 2])   #a2のx応力
    a2_sgmY = np.interp(a1_p, a2_pwork, a2[:, 3])   #a2のy応力
    # searched_a2 = [a2_epsX[0], a2_epsY[0], a2_sgmX[0], a2_sgmY[0]]  #各値がnp.array[]に入っているため外す
    searched_a2 = [a1_p[0], a2_epsX[0], a2_epsY[0], a2_sgmX[0], a2_sgmY[0]]  #各値がnp.array[]に入っているため外す
    return searched_a2

#等塑性仕事面の0degの値を検索
def search_values_00deg(a1: np.array, num) -> np.ndarray:   #(0deg, 検索値)
    a1_epsY = np.interp([num],  a1[:, 0], a1[:, 1])    #a1のyひずみ
    a1_sgmX = np.interp([num],  a1[:, 0], a1[:, 2])    #a1のx応力
    a1_sgmY = np.interp([num],  a1[:, 0], a1[:, 3])    #a1のy応力
    searched_a1 = [num, a1_epsY[0], a1_sgmX[0], a1_sgmY[0]] #各値がnp.array[]に入っているため外す
    return searched_a1

#output_listの作成
def create_output_list(dict1: dict, dict2: dict, eq_strains: list, deq_strain: float) -> list: #([検索する相当塑性ひずみ], ひずみ増分の差分値, 00deg, 他試験)
    for i in dict1.keys():
        dict1_key = i
    
    output_list = [['等塑性仕事面'], ['Δeq_plastic_strain=', deq_strain]]
    for i in eq_strains:
        output_list.append([])
        output_list.append(['eq_plastic_strain=', i])
        output_list.append(['ファイル名', 'plastic_work','plastic_strain_X', 'plastic_strain_Y', 'stress_X[MPa]', 'stress_Y[MPa]', 'Δstrain_Y/Δstrain_X'])
        a = search_values_00deg(dict1[dict1_key], i)                        #0degのひずみと応力の値を取得
        b = strain_increment_00deg(dict1[dict1_key], i, deq_strain)         #0degのひずみ増分方向の取得
        output_list.append([dict1_key.name] + [""] + a + [b])
        for j in dict2.keys():
            c = list(search_values(dict1[dict1_key], dict2[j], i))          #各試験のひずみと応力の値を取得
            d = strain_increment(dict1[dict1_key], dict2[j], i, deq_strain) #各試験のひずみ増分方向の取得
            c.append(d)
            c.insert(0, j.name)
            output_list.append(c)
    return output_list

def main():
    #パスの読み取り
    p_0, p_other = get_csv_paths('./ex_data_raw/')  #実験データを入れたデレクトリのパスを指定する
    #データの読み取り
    dict_0 = {i:read_data(i) for i in p_0}
    dict_other = {i:read_data(i) for i in p_other}

    eq_strains = [0.01, 0.02, 0.03,0.05,0.1,0.15,0.20,0.25,0.30]             #作成したい相当塑性ひずみを入力
    deq_strain = 0.005                          #ひずみ増分方向の差分の値を入力

    a = create_output_list(dict_0, dict_other, eq_strains, deq_strain)
    pprint(a)
    #csv書出し
    write_csv('contours.csv', a)               #結果を出力するcsvファイルの名前を指定する

if __name__ == '__main__':
    main()