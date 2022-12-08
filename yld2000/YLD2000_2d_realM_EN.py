import csv

from sympy import *


class MyAbs(Abs):
    def _eval_derivative(self, x):
        return Derivative(self.args[0], x, evaluate=True) * sign(
            conjugate(self.args[0])
        )


# csv書き込み関数
def write_csv(csv_name, data_list):
    with open(csv_name, "w", encoding="shift_jis", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(data_list)


# YLD2000-2dのパラメータ同定関数
def create_YLD2000_2d_parameter(
    sig_00, sig_45, sig_90, sig_b, r_00, r_45, r_90, r_b, M
):
    # ヤコビ行列と逆行列の生成関数
    def create_jac(def_list, var_list, subs_list):
        jac1 = []
        for i in def_list:
            ad1 = 0
            ad2 = []
            for j in var_list:
                ad1 = diff(i, j)
                ad2.append(ad1)
            jac1.append(ad2)
        jac_matrix = Matrix(jac1)
        jac_inv_matrix = jac_matrix.subs(subs_list).inv()
        return jac_inv_matrix

    # sympyを使ったニュートン-ラフソン法の関数
    def newton_raphson(var_list, var0_list, def_list):
        len_var = len(var_list)
        len_def = len(def_list)
        threshold = 1e-6  # 計算打ち切り閾値
        res = threshold + 1  # 残差(初期値は閾値より大きくしておく)
        n = 0
        a1 = [str(i) for i in var_list + def_list]  # hitryの列タイトル記入
        a1.insert(0, "error")
        a1.insert(0, "n(Number of calculations)")
        a1.insert(
            len(a1),
            [
                "sig_00=" + str(sig_00),
                "sig_45=" + str(sig_45),
                "sig_90=" + str(sig_90),
                "sig_b=" + str(sig_b),
                "r_00=" + str(r_00),
                "r_45=" + str(r_45),
                "r_90=" + str(r_90),
                "r_b=" + str(r_b),
                "M=" + str(M),
            ],
        )
        histry = [a1]
        var_matrix = Matrix(var_list)  # 変数の行列生成
        var0_matrix = Matrix(var0_list)  # 初期値の行列生成
        def_matrix = Matrix(def_list)  # 目的関数の行列生成
        x1 = zeros(len_var, 1)
        x2 = zeros(len_var, 1)
        x1 = x1 + var0_matrix

        subs_list = [(var_list[i], x1[i]) for i in range(len_var)]  # histryの初期値の記入
        b = x1.row_insert(len_var, def_matrix.subs(subs_list))
        b2 = b.T
        ad_histry = list(b2)
        ad_histry.insert(0, res)
        ad_histry.insert(0, n)
        histry.append(ad_histry)

        print(x1)
        # 残差が閾値より小さくなるまでループを回す
        while true:
            for i in range(len_var):
                jac_inv_def_matrix = (
                    create_jac(def_list, var_list, subs_list) * def_matrix
                )
                x2[i] = x1[i] - jac_inv_def_matrix[i].subs(subs_list)  # ニュートン-ラフソン法の漸化式
                # x2[i] = float(x1[i] - jac_inv_def_matrix[i].subs(subs_list))                # ニュートン-ラフソン法の漸化式
            print(x2)

            n += 1
            a = x2 - x1  # 2乗和の誤差計算
            res = sum([i**2 for i in a])
            subs_list2 = [(var_list[i], x2[i]) for i in range(len_var)]  # histryの記入
            b = x2.row_insert(len_var, def_matrix.subs(subs_list2))
            b2 = b.T
            ad_histry = list(b2)
            ad_histry.insert(0, res)
            ad_histry.insert(0, n)
            histry.append(ad_histry)

            print("res=", res)

            # 収束判定
            if res < threshold and min(x2) > 0:  # 成功時
                print("Success!")
                print(
                    "Number of calculations",
                    n,
                )
                # csv_write("histry_success_yld.csv", histry)
                converge_check = "true"
                break
            elif max([abs(i) for i in x2]) > 1e20:  # alphaが無限に発散する場合
                print("Failure! Parameters did not converge.")
                print(
                    "Number of calculations",
                    n,
                )
                # csv_write("histry_failure_yld.csv", histry)
                converge_check = "false"
                break
            elif n == 30:  # resが収束しない場合
                print("Failure! Error did not converge.")
                print(
                    "Number of calculations",
                    n,
                )
                # csv_write("histry_failure_yld.csv", histry)
                converge_check = "false"
                break

            x1 = zeros(len_var, 1)  # 変数の更新
            x1 = x1 + x2
            subs_list = [(var_list[i], x1[i]) for i in range(len_var)]

        return x2, converge_check

    # 変数宣言(sympy)
    alpha1 = Symbol("alpha1", real=True)
    alpha2 = Symbol("alpha2", real=True)
    alpha3 = Symbol("alpha3", real=True)
    alpha4 = Symbol("alpha4", real=True)
    alpha5 = Symbol("alpha5", real=True)
    alpha6 = Symbol("alpha6", real=True)
    alpha7 = Symbol("alpha7", real=True)
    alpha8 = Symbol("alpha8", real=True)
    sig_xx = Symbol("sig_xx", real=True)
    sig_yy = Symbol("sig_yy", real=True)
    sig_xy = Symbol("sig_xy", real=True)
    # 変数のリスト
    # alpha1~alpha6
    var_list = [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6]
    var0_list = [2, 2, 2, 2, 2, 2]  # 変数初期値
    # alpha7~alpha8
    var_list2 = [alpha7, alpha8]
    var0_list2 = [2, 2]  # 変数初期値

    # 目的関数の定義
    # YLD2000-2dの式記述
    # 同定式記述用の省略式
    # A1, A2, B1, B2
    A1 = (2 * alpha3 - 2 * alpha4 + 4 * alpha5 - alpha6) / 9
    A2 = (-alpha3 + 4 * alpha4 - 2 * alpha5 + 2 * alpha6) / 9
    B1 = (-6 * alpha3 + 6 * alpha4 + 12 * alpha5 - 3 * alpha6) / 9
    B2 = (3 * alpha3 - 12 * alpha4 - 6 * alpha5 + 6 * alpha6) / 9
    # alpha1 ~ alpha6
    g_0_1 = (
        sign(2 * alpha1 + alpha2)
        * pow(MyAbs(2 * alpha1 + alpha2), M - 1)
        * ((1 - r_00) * alpha1 + (r_00 + 2) * alpha2)
    )
    g_0_2 = (
        sign(2 * alpha3 - 2 * alpha4)
        * pow(MyAbs(2 * alpha3 - 2 * alpha4), M - 1)
        * ((1 - r_00) * alpha3 - (2 * r_00 + 4) * alpha4)
    )
    g_0_3 = (
        sign(4 * alpha5 - alpha6)
        * pow(MyAbs(4 * alpha5 - alpha6), M - 1)
        * ((2 - 2 * r_00) * alpha5 - (r_00 + 2) * alpha6)
    )

    g_90_1 = (
        sign(-alpha1 - 2 * alpha2)
        * pow(MyAbs(-alpha1 - 2 * alpha2), M - 1)
        * ((r_90 + 2) * alpha1 + (1 - r_90) * alpha2)
    )
    g_90_2 = (
        sign(-alpha3 + 4 * alpha4)
        * pow(MyAbs(-alpha3 + 4 * alpha4), M - 1)
        * ((r_90 + 2) * alpha3 + (2 * r_90 - 2) * alpha4)
    )
    g_90_3 = (
        sign(-2 * alpha5 + 2 * alpha6)
        * pow(MyAbs(-2 * alpha5 + 2 * alpha6), M - 1)
        * ((2 * r_90 + 4) * alpha5 - (1 - r_90) * alpha6)
    )

    g_b_1 = (
        sign(alpha1 - alpha2)
        * pow(MyAbs(alpha1 - alpha2), M - 1)
        * ((2 * r_b + 1) * alpha1 + (r_b + 2) * alpha2)
    )
    g_b_2 = (
        sign(alpha3 + 2 * alpha4)
        * pow(MyAbs(alpha3 + 2 * alpha4), M - 1)
        * ((2 * r_b + 1) * alpha3 - (2 * r_b + 4) * alpha4)
    )
    g_b_3 = (
        sign(2 * alpha5 + alpha6)
        * pow(MyAbs(2 * alpha5 + alpha6), M - 1)
        * ((4 * r_b + 2) * alpha5 - (r_b + 2) * alpha6)
    )
    # alpha7 ~ alpha8
    f_45_1 = pow(MyAbs(((alpha1 - alpha2) / 6) ** 2 + alpha7**2), M / 2)
    f_45_2 = pow(
        MyAbs(3 / 4 * (A1 + A2) - (((B1 + B2) / 2) ** 2 + alpha8**2) ** (1 / 2) / 2),
        M,
    )
    f_45_3 = pow(
        MyAbs(3 / 4 * (A1 + A2) + (((B1 + B2) / 2) ** 2 + alpha8**2) ** (1 / 2) / 2),
        M,
    )
    # Φ'
    phi1 = pow(
        MyAbs(
            ((2 * alpha1 + alpha2) * sig_xx + (-alpha1 - 2 * alpha2) * sig_yy) ** 2 / 9
            + 4 * (alpha7 * sig_xy) ** 2
        ),
        M / 2,
    )
    # Φ"
    phi2_1 = pow(
        MyAbs(
            3 / 2 * (A1 * sig_xx + A2 * sig_yy)
            - ((B1 * sig_xx + B2 * sig_yy) ** 2 + 4 * (alpha8 * sig_xy) ** 2) ** (1 / 2)
            / 2
        ),
        M,
    )
    phi2_2 = pow(
        MyAbs(
            3 / 2 * (A1 * sig_xx + A2 * sig_yy)
            + ((B1 * sig_xx + B2 * sig_yy) ** 2 + 4 * (alpha8 * sig_xy) ** 2) ** (1 / 2)
            / 2
        ),
        M,
    )
    phi2 = phi2_1 + phi2_2
    # Φ = Φ' + Φ"
    phi = phi1 + phi2
    # dΦ/dsig_xx, dΦ/dsig_yy
    phi_diff_sig_xx = diff(phi, sig_xx)
    phi_diff_sig_yy = diff(phi, sig_yy)
    # dΦ/dsig_xx, dΦ/dsig_yy (sigg_xx = sig_yy = sig_xy = sig_45/2)
    subs_list_45 = [(sig_xx, sig_45 / 2), (sig_yy, sig_45 / 2), (sig_xy, sig_45 / 2)]
    phi_diff_sig_xx_45 = phi_diff_sig_xx.subs(subs_list_45)
    phi_diff_sig_yy_45 = phi_diff_sig_yy.subs(subs_list_45)

    # 同定式の記述
    # alpha1 ~ alpha6
    f_0 = (
        pow(MyAbs(2 * alpha1 + alpha2), M)
        + pow(MyAbs(2 * alpha3 - 2 * alpha4), M)
        + pow(MyAbs(4 * alpha5 - alpha6), M)
        - 2 * pow(3 * sig_00 / sig_00, M)
    )
    f_90 = (
        pow(MyAbs(-alpha1 - 2 * alpha2), M)
        + pow(MyAbs(-alpha3 + 4 * alpha4), M)
        + pow(MyAbs(-2 * alpha5 + 2 * alpha6), M)
        - 2 * pow(3 * sig_00 / sig_90, M)
    )
    f_b = (
        pow(MyAbs(alpha1 - alpha2), M)
        + pow(MyAbs(alpha3 + 2 * alpha4), M)
        + pow(MyAbs(2 * alpha5 + alpha6), M)
        - 2 * pow(3 * sig_00 / sig_b, M)
    )
    g_0 = g_0_1 + g_0_2 + g_0_3
    g_90 = g_90_1 + g_90_2 + g_90_3
    g_b = g_b_1 + g_b_2 + g_b_3

    def_list = [f_0, f_90, f_b, g_0, g_90, g_b]

    # alpha7 ~ alpha8
    f_45 = f_45_1 + f_45_2 + f_45_3 - 2 * (sig_00 / sig_45) ** M
    g_45 = (
        phi_diff_sig_xx_45
        + phi_diff_sig_yy_45
        - 2 * M * sig_00**M / (sig_45 * (1 + r_45))
    )

    # ニュートン法の実行
    # alpha1~alpha6の同定
    y1, converge_check1 = newton_raphson(var_list, var0_list, def_list)

    alpha_subs_list = [
        (alpha1, y1[0]),
        (alpha2, y1[1]),
        (alpha3, y1[2]),
        (alpha4, y1[3]),
        (alpha5, y1[4]),
        (alpha6, y1[5]),
    ]
    # alpha1~alpha6の同定結果をf_45,g_45に代入
    f_45_def = f_45.subs(alpha_subs_list)
    g_45_def = g_45.subs(alpha_subs_list)
    def_list2 = [f_45_def, g_45_def]

    # alpha7~alpha8の同定
    if converge_check1 == "false":  # alpha1~alpha6の同定に失敗した場合，alpha7・alpha8を1000にする
        y2 = Matrix([1000, 1000])
        converge_check2 = "false"
    elif converge_check1 == "true":
        y2, converge_check2 = newton_raphson(var_list2, var0_list2, def_list2)

    alpha_list = [
        i for i in [j for j in y1] + [k for k in y2]
    ]  # alpha1~alpha8をMatrixからリストに変更

    return alpha_list, converge_check1, converge_check2


# ニュートン法の実行と同定結果の表示
def exe_create_YLD2000_2d_parameter(ex_value):

    alpha_list, converge_check1, converge_check2 = create_YLD2000_2d_parameter(
        *ex_value
    )  # alpha1~alpha8の同定

    # alpha1~alpha8を有効数字4桁に修正する関数
    def my_round(alpha_list):
        for j in range(len(alpha_list)):
            if alpha_list[j] < 10:
                alpha_list[j] = "{:.4g}".format(
                    alpha_list[j]
                )  # 10より小さいとき0.xxxx or x.xxx
            else:
                alpha_list[j] = "{:.3E}".format(alpha_list[j])  # 10以上のとき，x.xxxE+xx
        return alpha_list

    # 実験値と同定結果のprint
    print("------------------Identification Result------------------")

    if converge_check1 == "true" and converge_check2 == "true":
        print("Identification Success!")
        my_round(alpha_list)
    elif converge_check1 == "false" or converge_check2 == "false":
        print("Identification Failure!")

    print("sig_00, sig_45, sig_90, sig_b, r_00, r_45, r_90, r_b, M")
    print(
        ex_value[0],
        ex_value[1],
        ex_value[2],
        ex_value[3],
        ex_value[4],
        ex_value[5],
        ex_value[6],
        ex_value[7],
        ex_value[8],
    )
    print("alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8")
    print(
        alpha_list[0],
        alpha_list[1],
        alpha_list[2],
        alpha_list[3],
        alpha_list[4],
        alpha_list[5],
        alpha_list[6],
        alpha_list[7],
    )

    return ex_value, alpha_list


# 実験値の手入力関数
def input_ex_value():
    ex_value = [0 for i in range(9)]
    ex_value[0] = float(input("sig_00 = "))
    ex_value[1] = float(input("sig_45 = "))
    ex_value[2] = float(input("sig_90 = "))
    ex_value[3] = float(input("sig_b = "))
    ex_value[4] = float(input("r_00 = "))
    ex_value[5] = float(input("r_45 = "))
    ex_value[6] = float(input("r_90 = "))
    ex_value[7] = float(input("r_b = "))
    ex_value[8] = float(input("M = "))

    return ex_value


# 同定結果の書出し設定
def export_config(name, text, config):
    col_name1 = [
        "sig_00",
        "sig_45",
        "sig_90",
        "sig_b",
        "r_00",
        "r_45",
        "r_90",
        "r_b",
        "M",
    ]  # csvの列名
    col_name2 = [
        "alpha1",
        "alpha2",
        "alpha3",
        "alpha4",
        "alpha5",
        "alpha6",
        "alpha7",
        "alpha8",
    ]
    csv_list = [col_name1 + col_name2]
    csv_list.append(text)

    if config == "Y":
        write_csv(name, csv_list)

    return


# -------------------------------以下実行部--------------------------------
# 実行文
# mode="Input"
# mode = "As follows"
# ex_value=[1,1.038,1.020,1.064,2.477,1.762,2.500,0.93,4.27435]
# ex_value = [1, 1.0251, 0.9893, 1.2023, 2.1338, 1.5367, 2.2030, 0.8932, 4.5]

def export(ex_value):
    ex_value, alpha_list = exe_create_YLD2000_2d_parameter(ex_value)
    # 同定結果の書出し設定 (csvの名前, 変更しない, 'Y' or 'N')
    export_config(f"./yld2000/Parameters_yld2000-2d_test.csv", ex_value + alpha_list, "Y")


# if mode == "Input":
#     ex_value = input_ex_value()
def export_yld_parameter(ex_value,path,name):
    ex_value, alpha_list = exe_create_YLD2000_2d_parameter(ex_value)
    # 同定結果の書出し設定 (csvの名前, 変更しない, 'Y' or 'N')
    export_config(f"{path}/{name}.csv", ex_value + alpha_list, "Y")
