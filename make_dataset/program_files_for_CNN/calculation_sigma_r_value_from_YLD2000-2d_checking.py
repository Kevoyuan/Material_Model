# from asyncore import write
# from math import degrees
import sympy as sp
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import csv

class MyAbs(sp.Abs):
    def _eval_derivative(self, x):
        return sp.Derivative(self.args[0], x, evaluate=True)*sp.sign(sp.conjugate(self.args[0]))

#csv書き込み関数
def write_csv(csv_name, data_list):
    with open(csv_name,"w",encoding = "shift_jis",newline = "\n") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerows(data_list)

#応力の計算
#Yld2000-2dのX応力,Y応力の取得
def cal_sig_x_y(ratio, M, a1, a2, a3, a4, a5, a6, a7, a8):
    if ratio == 0:
        ratio = 1e-7
    # sig_x = (3*(2**(1/M))) / (((2*a1 + a2) + (-a1 -2*a2)*ratio)**M + ((2*a3 - 2*a4) + (-a3 + 4*a4)*ratio)**M + ((4*a5 -a6) + (-2*a5 + 2*a6)*ratio)**M)**(1/M)
    # sig_y = sig_x*ratio
    sig_x = (3*(2**(1/M))) / (abs((2*a1 + a2) + (-a1 -2*a2)*ratio)**M + abs((2*a3 - 2*a4) + (-a3 + 4*a4)*ratio)**M + abs((4*a5 -a6) + (-2*a5 + 2*a6)*ratio)**M)**(1/M)
    sig_y = ratio * sig_x
    return np.round([sig_x, sig_y], 3)

#ひずみ増分方向の計算
def cal_strain_increment(ratio, M, a1, a2, a3, a4, a5, a6, a7, a8):
    if ratio == 0:
        ratio = 1e-7
    # sigma_x > 0のときのみ正しい式，sigma_x < 0のときは符号が反転する
    k1 = sp.sign((2*a1 + a2) + (-a1 - 2*a2)*ratio)*pow(MyAbs((2*a1 + a2) + (-a1 - 2*a2)*ratio),(M - 1))
    k2 = sp.sign((2*a3 - 2*a4) + (-a3 + 4*a4)*ratio)*pow(MyAbs((2*a3 - 2*a4) + (-a3 + 4*a4)*ratio),(M - 1))
    k3 = sp.sign((4*a5 - a6) + (-2*a5 + 2*a6)*ratio)*pow(MyAbs((4*a5 - a6) + (-2*a5 + 2*a6)*ratio),(M - 1))
    rho = (k1*(-a1 -2*a2) + k2*(-a3 + 4*a4) + k3*(-2*a5 + 2*a6))/(k1*(2*a1 + a2) + k2*(2*a3 - 2*a4) + k3*(4*a5 - a6))

    return float(rho)

#ひずみ増分方向の勾配から角度に変換
def gradient_to_degree(ratio, gradient):
    #等二軸より上側の時
    if ratio > 45:
        #ひずみ増分方向が負の時
        if np.degrees(np.arctan(gradient)) < 0:
            #+180deg
            deg = np.round(np.degrees(np.arctan(gradient)) + 180, 3)
        else:
            deg = np.round(np.degrees(np.arctan(gradient)), 3)
    else:
        deg = np.round(np.degrees(np.arctan(gradient)), 3)
    return deg
# def gradient_to_degree(ratio, gradient):


#単軸応力の計算
def cal_unixial_stress(RD, M, a1, a2, a3, a4, a5, a6, a7, a8):
    RD = np.radians(RD)         #degからradに変更
    sig_bar = 1 #相当応力

    A1 = (2*a3 - 2*a4 + 4*a5 - a6)/9
    A2 = (-a3 + 4*a4 - 2*a5 + 2*a6)/9
    B1 = (-6*a3 + 6*a4 + 12*a5 - 3*a6)/9
    B2 = (3*a3 - 12*a4 - 6*a5 + 6*a6)/9

    phi1 = pow(MyAbs((1/9)*((2*a1 + a2)*np.cos(RD)**2 + (-a1 - 2*a2)*np.sin(RD)**2)**2 + 4*(a7*np.sin(RD)*np.cos(RD))**2), M/2)
    phi2_1 = pow(MyAbs(3/2*(A1*np.cos(RD)**2 + A2*np.sin(RD)**2) - (((B1*np.cos(RD)**2 + B2*np.sin(RD)**2)**2 + 4*(a8*np.sin(RD)*np.cos(RD))**2)**0.5)/2), M)
    phi2_2 = pow(MyAbs(3/2*(A1*np.cos(RD)**2 + A2*np.sin(RD)**2) + (((B1*np.cos(RD)**2 + B2*np.sin(RD)**2)**2 + 4*(a8*np.sin(RD)*np.cos(RD))**2)**0.5)/2), M)
    sig_phi = ((2*sig_bar**M)/(phi1 + phi2_1 + phi2_2))**(1/M)

    # return sig_phi
    return np.round(float(sig_phi), 3)

#r値の計算
def cal_r_value(sig_phi, RD, M, a1, a2, a3, a4, a5, a6, a7, a8):
    RD = np.radians(RD)         #degからradに変更
    sig_bar = 1 #相当応力

    sig_x = sp.Symbol('sig_x')
    sig_y = sp.Symbol('sig_y')
    sig_xy = sp.Symbol('sig_xy')

    A1 = (2*a3 - 2*a4 + 4*a5 - a6)/9
    A2 = (-a3 + 4*a4 - 2*a5 + 2*a6)/9
    B1 = (-6*a3 + 6*a4 + 12*a5 - 3*a6)/9
    B2 = (3*a3 - 12*a4 - 6*a5 + 6*a6)/9
    #Φ'
    phi1 = pow(MyAbs(((2 * a1 + a2) * sig_x + (- a1 - 2 * a2) * sig_y) ** 2 / 9 + 4 * (a7 * sig_xy) ** 2), M / 2)
    #Φ"
    phi2_1 = pow(MyAbs(3 / 2 * (A1 * sig_x + A2 * sig_y) - ((B1 * sig_x + B2 * sig_y) ** 2 + 4 * (a8 * sig_xy) ** 2) ** (1 / 2) / 2), M)
    phi2_2 = pow(MyAbs(3 / 2 * (A1 * sig_x + A2 * sig_y) + ((B1 * sig_x + B2 * sig_y) ** 2 + 4 * (a8 * sig_xy) ** 2) ** (1 / 2) / 2), M)
    phi2 = phi2_1 + phi2_2
    #Φ = Φ' + Φ"
    phi = phi1 + phi2
    #phiをsig_xとsig_yで偏微分
    phi_diff_x = phi.diff(sig_x)
    phi_diff_y = phi.diff(sig_y)

    #r値定義
    r_value = (2*M*sig_bar**M/sig_phi)/(phi_diff_x + phi_diff_y) - 1
    #座標変換の代入
    r_value = r_value.subs([(sig_x, sig_phi*np.cos(RD)**2), (sig_y, sig_phi*np.sin(RD)**2), (sig_xy, sig_phi*np.sin(RD)*np.cos(RD))])
    # return np.round(r_value, 2)
    return r_value



#描画
def draw(x, y):
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.title("Connected Scatterplot points with line")
    plt.xlabel("x")
    plt.ylabel(r'$\sigma_x/\sigma_0$')
    plt.show()



def exe_create_experimental_parameter(YLD2K_parameter):
    #Yld2000-2dパラメータ [M, a1, a2, ~ , a8]
    # parameter = [5.6,1.013,1.128,0.9059,0.9160,0.9192,0.8780,0.9830,0.8969]
    # parameter = [6.4, 1.023, 0.7723, 1.047, 0.867, 0.9678, 0.7325, 1.174, 1.707]
    # parameter = [5.328, 0.9500, 1.2212, 1.0396, 0.9098, 0.9339, 0.9269, 0.9514, 0.9735]
    parameter = YLD2K_parameter

    cal_point = [0,45,90]

    sig_x_y = []
    strain_increment = []
    strain_increment_deg = []
    r_value = []
    unixial_stress = []
    for i in cal_point:
        #二軸
        ratio = np.tan(np.radians(i))
        parameter1 = [ratio] + parameter
        #ひずみ増分方向(勾配)
        df = cal_strain_increment(*parameter1)
        strain_increment.append(df)
        #ひずみ増分方向(deg)
        strain_increment_deg.append(gradient_to_degree(i, df))
        #sig_x_yでの降伏曲面
        sig_x_y.append(cal_sig_x_y(*parameter1))

        #一軸
        parameter2 = [i] + parameter
        #単軸応力(iは圧延方向)
        sig_phi = cal_unixial_stress(*parameter2)
        unixial_stress.append(sig_phi)
        #r値(iは圧延方向)
        parameter3 = [sig_phi] + parameter2
        r_value.append(cal_r_value(*parameter3))

    # draw(cal_point, unixial_stress)

    #結果の出力
    a_unixial_stress = np.array(unixial_stress)
    a_r_value = np.array(r_value)
    a_sig_b = np.array([sig_x_y[1][1]])
    a_r_b = np.array([strain_increment[1]])
    a_M=np.array([parameter[0]])

    a_output = np.hstack([a_unixial_stress, a_sig_b, a_r_value, a_r_b, a_M]) 

    output = [['sig_00', 'sig_45', 'sig_90', 'sig_b', 'r_00', 'r_45', 'r_90', 'r_b', 'M']]
    output += [a_output.tolist()]

    return output


def read_parameter_success_csv(parameter_success_csv='parameter_success.csv'):
        #YLD2000-2dの計算結果や入力した実験値をリストに読み取り
    with open(parameter_success_csv,'r') as paraf :
        para_read = csv.reader(paraf)
        parameter_list = [a for a in para_read]
        
        parameter_list_for_keywrod = []
        n = 0
        for i in parameter_list:
            if n > 0:
                parameter_list_for_keywrod.append(i)
            n += 1

        #読み取ったリストからMとalpha1~8をfloat型として取り出し
        ex_value_list=[]
        alpha_list_original = []
        for i in parameter_list_for_keywrod:
            ex_value_list.append(i[1:10])
            alpha_list_original.append(i[10:18])

        ex_list = []
        #リストの文字列の実数化
        for i in ex_value_list:
            temp1 = []
            for j in i:
                a = float(j)
                temp1.append(a)
            ex_list.append(temp1)
            
        alo_list = []
        #リストの文字列の実数化
        for i in alpha_list_original:
            temp2 = []
            for j in i:
                a = float(j)
                temp2.append(a)
            alo_list.append(temp2)

        #keywordのNoを取り出し
        key_No = [i[0] for i in parameter_list_for_keywrod]

    return ex_list, alo_list, key_No

if __name__ == '__main__':
    YLD2K_parameter = [6.4, 1.023, 0.7723, 1.047, 0.867, 0.9678, 0.7325, 1.174, 1.707]
    
    #実行文

    parameter_success_csv = 'parameter_success.csv'                         #パラメータcsvファイル名
    ex_list, alo_list, key_No=read_parameter_success_csv(parameter_success_csv)


    pprint(key_No)
    pprint("----------------------------------------")
    pprint(ex_list)
    pprint("----------------------------------------")
    pprint(alo_list)

    # ex_value, alpha_list = exe_create_YLD2000_2d_parameter(input_ex_value())
    cal_ex_list=[]
    for i in range(len(key_No)):
        YLD2K_parameter=[ex_list[i][8]]+alo_list[i]
        # print(YLD2K_parameter)
        output = exe_create_experimental_parameter(YLD2K_parameter)[1:]
        
        ad_ad2 = []
        for j in output[0]:
            a = float(j)
            ad_ad2.append(a)
        cal_ex_list.append(ad_ad2)
    pprint("----------------------------------------")
    pprint(cal_ex_list)
   
    write_csv('./output_sigma_r-value_list.csv', cal_ex_list)