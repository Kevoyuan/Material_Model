import os
import pprint
import subprocess
from multiprocessing import Pool
import csv
import time
from tqdm import tqdm
from numpy import insert

import datetime

#csv書き込み関数
def write_csv(csv_name, data_list):
    with open(csv_name,"w",encoding = "shift_jis",newline = "\n") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerows(data_list)

#各keywordファイルのファイル名とパスを取得
def cre_key_list():
    #parameter_success.csvからkeywordファイルの番号を読み取り
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
        #keywordファイルの番号を取得
        key_No = [i[0] for i in parameter_list_for_keywrod]
        key_name = [i[1] for i in parameter_list_for_keywrod]

        return key_No,key_name

    key_No,key_name = read_parameter(".././roughmodel_test/log/unfinished_files.csv")

    #keywordファイルの名前一覧の取得
    key_list = []

    for i in key_name:
        a = [i.split("\\")[-1] + ".k"
        , os.getcwd()+"\\"+i.split("\\",2)[2]
        ]

        # print(a)


        key_list.append(a)

    return key_list #[keywordファイル名, keywordフォルダのpath]

#実行部
def dyna_run(key_list) :
    #dyna_path = "D:\LSDYNA\program\ls-dyna_smp_s_R10_2_0_winx64_ifort160.exe" 
    NCPU="2"
    os.chdir(key_list[1])
    print(key_list[0],"を解析します")
    subprocess.run(key_list[2] + " I=" + key_list[0] +" NCPU=" + NCPU,shell = True)

    print("解析終了:",key_list[0])
    return [key_list[0]]

#実行関数
def P4(cpus=2, solver_path='D:\LSDYNA\program\ls-dyna_smp_d_R12.1_winx64_ifort170.exe'):   #(cpu数)
    start_time = time.time()
    key_list = cre_key_list()   #keywordファイル名とkeywordのpathの取得
    key_list = [i + [solver_path] for i in key_list]    #LS-DYNAのソルバーパスの追加

    #開始確認
    print("analyze list")
    pprint.pprint(key_list)
    print("use cpus :",cpus)
    # #解析実行
    # with Pool(cpus) as p :
    #     result = p.map(dyna_run,key_list,1)
    # pprint.pprint(key_list)

    # Open a log file for writing
    # log_filename = "simulation_log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
    # log_file = open(log_filename, "w")

    # Display a progress bar using tqdm
    with tqdm(total=len(key_list)) as pbar:
        #解析実行
        with Pool(cpus) as p :
            result = p.map(dyna_run,key_list)
            pbar.update()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Close the log file
    # log_file.close()



    print("finish")
    end_time = time.time()
    print("総解析時間[s]:", int(end_time - start_time))
    os.makedirs('./log', exist_ok = True)
    output_list = [["LS-DYNA総解析時間[s]", int(end_time - start_time)],
                   ['LS-DYNAソルバー名', solver_path],
                   ['解析ファイル数', len(key_list)]]
    write_csv('./log/ls-dyna_total_time.csv', output_list)

#実行部
def lsprepost_run(key_list) :
    #D:\LSDYNA\program\lsprepost.exe -nographics c=C:\WorkPy\lspostcmd.cfile

    os.chdir(key_list[1])
    print(key_list[0],"をデータ処理します")
    subprocess.run(key_list[2] + " -nographics c=" + key_list[3],shell = True)

    print("データ処理終了:",key_list[0])
    return [key_list[0]]

#実行関数
def P4_2(cpus=8, solver_path='D:\LSDYNA\program\lsprepost.exe', 
                 cfile_path='..\..\..\..\optional_files\lspostcmd.cfile'):   #(cpu数)
    start_time = time.time()
    key_list = cre_key_list()   #keywordファイル名とkeywordのpathの取得
    key_list = [i + [solver_path] + [cfile_path] for i in key_list]    #LS-PREPOSTのパス，cfileのパス追加

    #開始確認
    print("analyze list")
    pprint.pprint(key_list)
    print("use cpus :",cpus)
    #解析実行
    with Pool(cpus) as p :
        result = p.map(lsprepost_run,key_list,1)
    pprint.pprint(key_list)

    print("finish")
    end_time = time.time()
    print("総データ処理時間[s]:", int(end_time - start_time))
    os.makedirs('./log', exist_ok = True)
    output_list = [["LS-PREPOST総dデータ処理時間[s]", int(end_time - start_time)],
                   ['LS-PREPOST名', solver_path],
                   ['cfile名', cfile_path],
                   ['処理ファイル数', len(key_list)]]
    write_csv('./log/lsprepost_total_time.csv', output_list)
    
#以下実行部
if __name__ == '__main__':
    #P4()
    P4_2()
