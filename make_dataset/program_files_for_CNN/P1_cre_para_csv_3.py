import csv
import sympy as sp
import time
import math
import os
from tqdm import tqdm
import random

class MyAbs(sp.Abs):
    def _eval_derivative(self, x):
        return sp.Derivative(self.args[0], x, evaluate=True)*sp.sign(sp.conjugate(self.args[0]))

#ランダムに実験値を与えYLDのparameter.csvを作成する

#csv書き込み関数
def write_csv(csv_name, data_list):
    with open(csv_name,"w",encoding = "shift_jis",newline = "\n") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerows(data_list)

#実験値の生成関数
def create_ex_value(sig45_range, sig90_range, sigb_range, r00_range, r45_range, r90_range, rb_range, M_range):
    #レンジ内のランダムな値をfloatタイプで生成
    def create_random_ex(range, significant_figures):
        ex_value = round(random.uniform(range[0], range[1]), significant_figures)
        return ex_value

    sig_00 = 1
    sig_45 = create_random_ex(sig45_range, 3)
    sig_90 = create_random_ex(sig90_range, 3)
    sig_b = create_random_ex(sigb_range, 3)
    r_00 = create_random_ex(r00_range, 2)
    r_45 = create_random_ex(r45_range, 2)
    r_90 = create_random_ex(r90_range, 2)
    r_b = create_random_ex(rb_range, 3)
    M = create_random_ex(M_range, 3)
    # M = random.choice(M_range)
    
    a = [sig_00, sig_45, sig_90, sig_b, r_00, r_45, r_90, r_b, M]
    return a

#YLD2000-2dのパラメータ同定関数
def create_YLD2000_2d_parameter(sig_00, sig_45, sig_90, sig_b, r_00, r_45, r_90, r_b, M):
    #ヤコビ行列と逆行列の生成関数
    def create_jac(def_list, var_list, subs_list):
        jac1 = []
        for i in def_list:
            ad1 = 0
            ad2= []
            for j in var_list:
                ad1 = sp.diff(i, j)
                ad2.append(ad1)
            jac1.append(ad2)
        jac_matrix = sp.Matrix(jac1)
        jac_inv_matrix = jac_matrix.subs(subs_list).inv()
        return jac_inv_matrix

    # sympyを使ったニュートン-ラフソン法の関数
    def newton_raphson(var_list, var0_list, def_list):
        len_var = len(var_list)
        len_def = len(def_list)
        threshold = 1e-6                                                                    # 計算打ち切り閾値
        res = threshold + 1                                                                 # 残差(初期値は閾値より大きくしておく)
        n = 0
        a1 = [str(i) for i in var_list + def_list]                                          # hitryの列タイトル記入
        a1.insert(0, 'error')
        a1.insert(0, 'n(計算回数)')
        a1.insert(len(a1), ['sig_00='+ str(sig_00), 'sig_45='+ str(sig_45), 'sig_90='+ str(sig_90), 'sig_b='+ str(sig_b), 
                        'r_00=' + str(r_00), 'r_45=' + str(r_45), 'r_90=' + str(r_90), 'r_b=' + str(r_b), 'M=' + str(M)])
        histry = [a1]
        var_matrix = sp.Matrix(var_list)                                                       # 変数の行列生成
        var0_matrix = sp.Matrix(var0_list)                                                     # 初期値の行列生成
        def_matrix = sp.Matrix(def_list)                                                       # 目的関数の行列生成
        x1 = sp.zeros(len_var, 1)
        x2 = sp.zeros(len_var, 1)
        x1 = x1 + var0_matrix

        subs_list = [(var_list[i], x1[i]) for i in range(len_var)]                          # histryの初期値の記入
        b = x1.row_insert(len_var, def_matrix.subs(subs_list))
        b2 = b.T
        ad_histry = list(b2)
        ad_histry.insert(0, res)
        ad_histry.insert(0, n)
        histry.append(ad_histry)

        #print(x1)
        # 残差が閾値より小さくなるまでループを回す
        while True:
            for i in range(len_var):
                jac_inv_def_matrix = create_jac(def_list, var_list, subs_list) * def_matrix
                x2[i] = x1[i] - jac_inv_def_matrix[i].subs(subs_list)                # ニュートン-ラフソン法の漸化式
            #print(x2)
            
            n += 1
            a = x2 - x1                                                                     # 2乗和の誤差計算
            res = sum([i ** 2 for i in a])
            subs_list2 = [(var_list[i], x2[i]) for i in range(len_var)]                     # histryの記入
            b = x2.row_insert(len_var, def_matrix.subs(subs_list2))
            b2 = b.T
            ad_histry = list(b2)
            ad_histry.insert(0, res)
            ad_histry.insert(0, n)
            histry.append(ad_histry)

            #print('res=', res)

            #収束判定
            if res < threshold and min(x2) > 0:                                             # 成功時
                #print('Success!')
                #print('計算回数',n,)
                #csv_write("histry_success_yld.csv", histry)
                converge_check =  'true'
                break
            elif max([abs(i) for i in x2]) > 1e6:                                           # alphaが無限に発散する場合
                #print('Failure!　パラメータが収束しませんでした')
                #print('計算回数',n,)
                #csv_write("histry_failure_yld.csv", histry)
                converge_check =  'false'
                break
            elif n == 30:                                                                   # resが収束しない場合
                #print('Failure!　誤差が収束しませんでした')
                #print('計算回数',n,)
                #csv_write("histry_failure_yld.csv", histry)
                converge_check =  'false'
                break
            
            x1 = sp.zeros(len_var, 1)                                                          # 変数の更新
            x1 = x1 + x2
            subs_list = [(var_list[i], x1[i]) for i in range(len_var)]

        return x2, converge_check
    
    #変数宣言(sympy)
    alpha1 = sp.Symbol('alpha1', real=True)
    alpha2 = sp.Symbol('alpha2', real=True)
    alpha3 = sp.Symbol('alpha3', real=True)
    alpha4 = sp.Symbol('alpha4', real=True)
    alpha5 = sp.Symbol('alpha5', real=True)
    alpha6 = sp.Symbol('alpha6', real=True)
    alpha7 = sp.Symbol('alpha7', real=True)
    alpha8 = sp.Symbol('alpha8', real=True)
    sig_xx = sp.Symbol('sig_xx', real=True)
    sig_yy = sp.Symbol('sig_yy', real=True)
    sig_xy = sp.Symbol('sig_xy', real=True)
    #変数のリスト
    #alpha1~alpha6
    var_list = [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6]
    var0_list = [2, 2, 2, 2, 2, 2]                                                          # 変数初期値
    #alpha7~alpha8
    var_list2 = [alpha7, alpha8]          
    var0_list2 = [2, 2]                                                                     # 変数初期値

    #目的関数の定義
    #YLD2000-2dの式記述
    #同定式記述用の省略式
    #A1, A2, B1, B2
    A1 = (2 * alpha3 - 2 * alpha4 + 4 * alpha5 - alpha6) / 9
    A2 = (- alpha3 + 4 * alpha4 - 2 * alpha5 + 2 * alpha6) / 9
    B1 = (- 6 * alpha3 + 6 * alpha4 + 12 * alpha5 - 3 * alpha6) / 9
    B2 = (3 * alpha3 - 12 * alpha4 - 6 * alpha5 + 6 * alpha6) / 9
    # alpha1 ~ alpha6
    g_0_1 = sp.sign(2 * alpha1 + alpha2)*pow(MyAbs(2 * alpha1 + alpha2), M - 1) * ((1 - r_00) * alpha1 + (r_00 + 2) * alpha2)
    g_0_2 = sp.sign(2 * alpha3 - 2 * alpha4)*pow(MyAbs(2 * alpha3 - 2 * alpha4), M - 1) * ((1 - r_00) * alpha3 - (2 * r_00 + 4) * alpha4)
    g_0_3 = sp.sign(4 * alpha5 - alpha6)*pow(MyAbs(4 * alpha5 - alpha6), M - 1) * ((2 - 2 * r_00) * alpha5 - (r_00 + 2) * alpha6)

    g_90_1 = sp.sign(- alpha1 - 2 * alpha2)*pow(MyAbs(- alpha1 - 2 * alpha2), M - 1) * ((r_90 + 2) * alpha1 + (1 - r_90) * alpha2)
    g_90_2 = sp.sign(- alpha3 + 4 * alpha4)*pow(MyAbs(- alpha3 + 4 * alpha4), M - 1) * ((r_90 + 2) * alpha3 + (2 * r_90 - 2) * alpha4)
    g_90_3 = sp.sign(- 2 * alpha5 + 2 * alpha6)*pow(MyAbs(- 2 * alpha5 + 2 * alpha6), M - 1) * ((2 * r_90 + 4) * alpha5 - (1 - r_90) * alpha6)

    g_b_1 = sp.sign(alpha1 - alpha2)*pow(MyAbs(alpha1 - alpha2), M -1) * ((2 * r_b + 1) * alpha1 + (r_b + 2) * alpha2)
    g_b_2 = sp.sign(alpha3 + 2 * alpha4)*pow(MyAbs(alpha3 + 2 * alpha4), M -1) * ((2 * r_b + 1) * alpha3 - (2 * r_b + 4) * alpha4)
    g_b_3 = sp.sign(2 * alpha5 + alpha6)*pow(MyAbs(2 * alpha5 + alpha6), M -1) * ((4 * r_b + 2) * alpha5 - (r_b + 2) * alpha6)
    # alpha7 ~ alpha8
    f_45_1 = pow(MyAbs(((alpha1 - alpha2) / 6) ** 2 + alpha7 ** 2), M / 2)
    f_45_2 = pow(MyAbs(3 / 4 * (A1 + A2) - (((B1 + B2) / 2) ** 2 + alpha8 ** 2) ** (1/2) / 2), M)
    f_45_3 = pow(MyAbs(3 / 4 * (A1 + A2) + (((B1 + B2) / 2) ** 2 + alpha8 ** 2) ** (1/2) / 2), M)
    #Φ'
    phi1 = pow(MyAbs(((2 * alpha1 + alpha2) * sig_xx + (- alpha1 - 2 * alpha2) * sig_yy) ** 2 / 9 + 4 * (alpha7 * sig_xy) ** 2), M / 2)
    #Φ"
    phi2_1 = pow(MyAbs(3 / 2 * (A1 * sig_xx + A2 * sig_yy) - ((B1 * sig_xx + B2 * sig_yy) ** 2 + 4 * (alpha8 * sig_xy) ** 2) ** (1 / 2) / 2), M)
    phi2_2 = pow(MyAbs(3 / 2 * (A1 * sig_xx + A2 * sig_yy) + ((B1 * sig_xx + B2 * sig_yy) ** 2 + 4 * (alpha8 * sig_xy) ** 2) ** (1 / 2) / 2), M)
    phi2 = phi2_1 + phi2_2
    #Φ = Φ' + Φ"
    phi = phi1 + phi2
    #dΦ/dsig_xx, dΦ/dsig_yy 
    phi_diff_sig_xx = sp.diff(phi, sig_xx)
    phi_diff_sig_yy = sp.diff(phi, sig_yy)
    #dΦ/dsig_xx, dΦ/dsig_yy (sigg_xx = sig_yy = sig_xy = sig_45/2)
    subs_list_45 = [(sig_xx, sig_45 / 2), (sig_yy, sig_45 / 2), (sig_xy, sig_45 / 2)]
    phi_diff_sig_xx_45 = phi_diff_sig_xx.subs(subs_list_45)
    phi_diff_sig_yy_45 = phi_diff_sig_yy.subs(subs_list_45)

    #同定式の記述
    # alpha1 ~ alpha6
    f_0 = pow(MyAbs(2 * alpha1 + alpha2), M) + pow(MyAbs(2 * alpha3 - 2 * alpha4), M) + pow(MyAbs(4 * alpha5 - alpha6), M) - 2 * pow(3 * sig_00 / sig_00, M)
    f_90 = pow(MyAbs(- alpha1 - 2 * alpha2), M) + pow(MyAbs(- alpha3 + 4 * alpha4), M) + pow(MyAbs(- 2 * alpha5 + 2 * alpha6), M) - 2 * pow(3 * sig_00 / sig_90, M)
    f_b = pow(MyAbs(alpha1 - alpha2), M) + pow(MyAbs(alpha3 + 2 * alpha4), M) + pow(MyAbs(2 * alpha5 + alpha6), M) - 2 * pow(3 * sig_00 / sig_b, M)
    g_0 =  g_0_1 + g_0_2 + g_0_3
    g_90 = g_90_1 + g_90_2 + g_90_3
    g_b = g_b_1 + g_b_2 + g_b_3

    def_list = [f_0, f_90, f_b, g_0, g_90, g_b]

    # alpha7 ~ alpha8
    f_45 = f_45_1 + f_45_2 + f_45_3 - 2 * (sig_00 / sig_45) ** M
    g_45 = phi_diff_sig_xx_45 + phi_diff_sig_yy_45 - 2 * M * sig_00 ** M / (sig_45 * (1 + r_45))

    #ニュートン法の実行
    #alpha1~alpha6の同定
    y1, converge_check1 = newton_raphson(var_list, var0_list,  def_list)

    alpha_subs_list = [(alpha1, y1[0]), (alpha2, y1[1]), (alpha3, y1[2]), (alpha4, y1[3]), (alpha5, y1[4]), (alpha6, y1[5])]
    #alpha1~alpha6の同定結果をf_45,g_45に代入
    f_45_def = f_45.subs(alpha_subs_list)
    g_45_def = g_45.subs(alpha_subs_list)
    def_list2 = [f_45_def, g_45_def]

    #alpha7~alpha8の同定
    if converge_check1 == 'false' :                                                                  # alpha1~alpha6の同定に失敗した場合，alpha7・alpha8を1000にする
        y2 = sp.Matrix([1000, 1000])
        converge_check2 = 'false'
    elif converge_check1 ==  'true' :
        y2, converge_check2 = newton_raphson(var_list2, var0_list2,  def_list2)

    alpha_list = [i for i in [j for j in y1] + [k for k in y2]]                                     # alpha1~alpha8をMatrixからリストに変更

    return alpha_list, converge_check1, converge_check2

#ランダム生成した実験値によるパラメータ同定関数
def P1(end_count, sig45_range, sig90_range, sigb_range, r00_range, r45_range, r90_range, rb_range, M_range):
    pbar = tqdm(total=end_count)                                                                    # プログレスバーの最大値の設定
    start_time = time.time()                                                                        # 開始時刻の記録
    col_name1 = ['n', 'sig_00', 'sig_45', 'sig_90', 'sig_b', 'r_00', 'r_45', 'r_90', 'r_b', 'M']    # csvの列名
    col_name2 = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6', 'alpha7', 'alpha8']
    csv_list_success = [col_name1 + col_name2]
    csv_list_failure = [col_name1 + col_name2]
    n = 1
    n_success = 0
    n_failure = 0
    ##alpha1~alpha8を有効数字4桁に修正する関数
    def my_round(alpha_list):
        for j in range(len(alpha_list)):
            if alpha_list[j] < 10:
                alpha_list[j] = '{:.4g}'.format(alpha_list[j])                                      # 10より小さいとき0.xxxx or x.xxx
            else:
                alpha_list[j] = '{:.3E}'.format(alpha_list[j])                                      # 10以上のとき，x.xxxE+xx
        return alpha_list

    while True:
        #ランダムな実験値でのalpha1~alpha8の同定
        #ランダム生成する実験値の範囲(sig00は1で固定)
        ex_value = create_ex_value(sig45_range, sig90_range, sigb_range, r00_range, r45_range, r90_range, rb_range, M_range) # (sig45, sig90, sigb, r0, r45, r90, rb, M)
        #ニュートン法の実行 -> アルファのリスト, a1~a6の収束結果, a7~a8の収束結果 (true or false)
        alpha_list, converge_check1, converge_check2 = create_YLD2000_2d_parameter(*ex_value)
        
        if converge_check1 == 'true' and converge_check2 == 'true':                                 # 同定成功時はcsv_list_successに追加
            alpha_list = my_round(alpha_list)
            ad_csv_list = [n_success] + ex_value + alpha_list
            csv_list_success.append(ad_csv_list)
            n_success += 1
            pbar.update(1)                                                                          # ステータスバーの更新

        elif converge_check1 == 'false' or converge_check2 == 'false':                              # 同定失敗時はcsv_list_failureに追加
            ad_csv_list = [n_failure] + ex_value + alpha_list
            csv_list_failure.append(ad_csv_list)
            n_failure += 1
        
        if n_success >= end_count:
            break
        n += 1

    pbar.close()                                                                                    #プログレスバーの終了
    write_csv('parameter_success.csv', csv_list_success)
    write_csv('parameter_failure.csv', csv_list_failure)
    process_time = time.time() - start_time                                                         # 経過時間の計算
    print('計算にかかった時間:', math.floor(process_time), '秒')
    print('総パラメータ数:', n)
    print('成功パラメータ数:', n_success)
    print('失敗パラメータ数:', n_failure)

    #ログ書出し
    os.makedirs('./log', exist_ok = True)
    log_list = [['計算にかかった時間[秒]', math.floor(process_time)],
                ['総パラメータ数', n],
                ['成功パラメータ数', n_success],
                ['失敗パラメータ数', n_failure],
                ['Mの範囲', str(M_range)],
                ['sig00の値', 1],
                ['sgm45の範囲', str(sig45_range)],
                ['sgm90の範囲', str(sig90_range)],
                ['sgmbの範囲', str(sigb_range)],
                ['r00の範囲', str(r00_range)],
                ['r45の範囲', str(r45_range)],
                ['r90の範囲', str(r90_range)],
                ['rbの範囲', str(rb_range)]]
    write_csv('./log/parameter_log.csv', log_list)
    return

#-------------------------------以下実行部--------------------------------
if __name__ == '__main__':
    #実行文　                           (作成する同定成功パラメータ数)
    P1(5)
