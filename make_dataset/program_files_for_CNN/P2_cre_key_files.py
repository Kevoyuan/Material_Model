import csv
import os

#parameterファイルを読み取ってkeywordファイルを作成する

def read_parameter(parameter_success_csv):
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

    #読み取ったリストからalpha1~8を取り出し
    alpha_list_for_keyword = []
    for i in range(len(parameter_list_for_keywrod)):
        alpha_list_for_keyword.append(parameter_list_for_keywrod[i][10:18])

    #alpha1~8ををkeywordに書き込めるように空白で10カラムに加工
    blunk_10 = '          '
    for i in range(len(alpha_list_for_keyword)):
        for j in range(8):
            alpha_list_for_keyword[i][j] = blunk_10 + alpha_list_for_keyword[i][j]
            alpha_list_for_keyword[i][j] = alpha_list_for_keyword[i][j][-10:]
        alpha_list_for_keyword[i] = ''.join(alpha_list_for_keyword[i])

    return parameter_list_for_keywrod, alpha_list_for_keyword

def write_keyword(parameter_list_for_keywrod, alpha_list_for_keyword, main_k_name, RD_list):

    '--------------------------'
    #y方向引張解析時の0,45,90度の圧延,圧延直角方向ベクトル
    RD_vector_0 = '                               0.0000000 1.0000000 0.0000000\n'
    TD_vector_0 = '                               1.0000000 0.0000000 0.0000000\n'
    RD_vector_45 = '                               1.0000000 1.0000000 0.0000000\n'
    TD_vector_45 = '                              -1.0000000 1.0000000 0.0000000\n'
    RD_vector_90 = '                               1.0000000 0.0000000 0.0000000\n'
    TD_vector_90 = '                               0.0000000 1.0000000 0.0000000\n'
    '---------------------------'

    n = 0
    for i in parameter_list_for_keywrod:
        for j in RD_list:
            os.makedirs('./keyword/keyword' + str(i[0]) + '/keyword'+ str(i[0]) + '_' + str(j), exist_ok = True)
            with open('./keyword/keyword'+ str(i[0]) + '/keyword'+ str(i[0]) + '_' + str(j) + '/keyword' +\
                str(i[0]) + '_' + str(j) + '.k', 'w', encoding = 'shift_jis', newline = '\n') as f:
                f.write('*KEYWORD\n')
                f.write('*TITLE\n')
                f.write('LS-DYNA keyword deck by LS-PrePost\n')
                f.write('*MAT_BARLAT_YLD2000\n')
                f.write('$SPCE_senda_direct\n')
                f.write('$      mid        ro         e        pr       fit      beta      iter\n')
                f.write('         1 7.830e-09 206000.00 0.3000000 0.0000000 0.0000000 0.0000000\n')
                f.write('$        k        e0         n         c         p      hard         a\n')
                f.write('                                                   -2.000000     ' + str(f"{float(i[9]):.3f}")+ '\n')
                f.write('$   alpha1    alpha2    alpha3    alpha4    alpha5    alpha6    alpha7    alpha8\n')
                f.write(str(alpha_list_for_keyword[n]) + '\n')
                f.write('$     aopt\n')
                f.write(' 2.0000000\n')

                if j == 0:    
                    f.write('$                                     a1        a2        a3\n')
                    f.write(RD_vector_0)
                    f.write('$       v1        v2        v3        d1        d2        d3\n')
                    f.write(TD_vector_0)
                elif j == 45:
                    f.write('$                                     a1        a2        a3\n')
                    f.write(RD_vector_45)
                    f.write('$       v1        v2        v3        d1        d2        d3\n')
                    f.write(TD_vector_45)
                elif j == 90:
                    f.write('$                                     a1        a2        a3\n')
                    f.write(RD_vector_90)
                    f.write('$       v1        v2        v3        d1        d2        d3\n')
                    f.write(TD_vector_90)

                f.write('*include\n')
                f.write(main_k_name + '\n')
                f.write('*END')
        n += 1

def P2():
    parameter_csv_name = 'parameter_success.csv'                        #パラメータcsvファイル名
    a, b = read_parameter(parameter_csv_name)                           #読み取る試験条件やYLD2000-2dのパラメータのcsvファイル名を入力

    # RD_list = [0, 45, 90]                                               #圧延方向[0, 45, 90 deg]
    RD_list = [0, 90]                                               #圧延方向[0, 45, 90 deg]
    
    write_keyword(a, b,'../../../../optional_files/main_explicit.k', RD_list)                              

'--------------------------------------------------------------------'
#実行部
if __name__ == '__main__':
    P2()        #mainのkeyeordファイル名，読み取るパラメータcsvファイル名をに入力
