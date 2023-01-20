import os
import pprint
import subprocess
from multiprocessing import Pool
import csv
import time

from numpy import insert

# function to write to a CSV file
def write_csv(csv_name, data_list):
    with open(csv_name,"w",encoding = "shift_jis",newline = "\n") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerows(data_list)

# function to get the names and paths of each keyword file
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

        return key_No

    key_No = read_parameter("parameter_success.csv")

    #keywordファイルの名前一覧の取得
    key_list = []
    for i in key_No:
        # for j in ["_0", "_45", "_90"]:
        for j in ["_0", "_90"]:

            # a = ["keyword" + i + j + ".k", os.getcwd() + "/keyword/keyword" + i + "/keyword" + i + j]
            a = ["keyword" + i + j + ".k", os.getcwd() + "\\keyword\\keyword" + i + "\\keyword" + i + j]
            key_list.append(a)

    return key_list #[keywordファイル名, keywordフォルダのpath]

#実行部
def dyna_run(key_list) :
    #dyna_path = "D:\LSDYNA\program\ls-dyna_smp_s_R10_2_0_winx64_ifort160.exe" 
    NCPU="2"
    os.chdir(key_list[1])
    print(key_list[0],"is being analyzed")
    subprocess.run(key_list[2] + " I=" + key_list[0] +" NCPU=" + NCPU,shell = True)

    print("Analysis completed:",key_list[0])
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
    #解析実行
    with Pool(cpus) as p :
        result = p.map(dyna_run,key_list,1)
    pprint.pprint(key_list)

    print("finish")
    end_time = time.time()
    print("Total analysis time [s]:", int(end_time - start_time))
    os.makedirs('./log', exist_ok = True)
    output_list = [["LS-DYNA total analysis time[s]", int(end_time - start_time)],
                        ['LS-DYNA solver name', solver_path],
                        ['number of analysis files', len(key_list)]]
    write_csv('./log/ls-dyna_total_time.csv', output_list)

#実行部
def lsprepost_run(key_list) :
    #D:\LSDYNA\program\lsprepost.exe -nographics c=C:\WorkPy\lspostcmd.cfile

    os.chdir(key_list[1])
    print(key_list[0], "process data")
    subprocess.run(key_list[2] + " -nographics c=" + key_list[3], shell = True)

    print("End of data processing:",key_list[0])
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
    print("Total data processing time [s]:", int(end_time - start_time))
    os.makedirs('./log', exist_ok = True)
    output_list = [["LS-PREPOST total d data processing time [s]", int(end_time - start_time)],
                ['LS-PREPOST name', solver_path],
                ['cfile name', cfile_path],
                ['Number of processed files', len(key_list)]]
    write_csv('./log/lsprepost_total_time.csv', output_list)
    
#以下実行部
if __name__ == '__main__':
    #P4()
    P4_2()
