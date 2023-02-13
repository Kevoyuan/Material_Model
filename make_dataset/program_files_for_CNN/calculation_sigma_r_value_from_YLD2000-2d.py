# from asyncore import write
# from math import degrees
import sympy as sp
import numpy as np
# from pprint import pprint
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
    return np.round(float(r_value), 3)



#描画
def draw(x, y,xlabel,ylabel):
    plt.scatter(x, y)
    plt.plot(x, y)
    # plt.title("Connected Scatterplot points with line")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()



def main():
    #Yld2000-2dパラメータ [M, a1, a2, ~ , a8]
    parameter = [6.4, 1.023, 0.7723, 1.047, 0.867, 0.9678, 0.7325, 1.174, 1.707]

    cal_point = [angle for angle in range(0,91,5)] # Angles from rolling direction 
    # cal_point = [0,22.5,45,67.5,90]
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

    draw(cal_point, unixial_stress,"Angle from rolling direction",r'$\sigma_x/\sigma_0$')
    draw(cal_point, r_value,"Angle from rolling direction","r-value")

    #結果の出力
    a_unixial_stress = np.array(unixial_stress)
    a_r_value = np.array(r_value)
    a_sig_b = np.array([sig_x_y[cal_point.index(45)][1]])
    a_r_b = np.array([strain_increment[cal_point.index(45)]])
    a_M=np.array([parameter[0]])

    a_output = np.hstack([a_unixial_stress, a_sig_b, a_r_value, a_r_b, a_M]) 

    # output = [['sig_00', 'sig_45', 'sig_90', 'sig_b', 'r_00', 'r_45', 'r_90', 'r_b', 'M']]
    output_header = ['sig_'+str(n) for n in cal_point]
    output_header.append('sig_b')
    output_header2= ['r_'+str(n) for n in cal_point]
    output_header2.extend(['r_b','M'])
    output_header += output_header2
    output =[output_header]+ [a_output.tolist()]
    print(output)
    write_csv('./output_sigma_r-value.csv', output)
    return


if __name__ == '__main__':
    main()
