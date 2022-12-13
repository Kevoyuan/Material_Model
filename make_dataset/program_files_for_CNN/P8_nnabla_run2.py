import pprint
import subprocess
import csv
from pprint import pprint
from pathlib import Path
import pandas as pd


#csv書き込み関数
def write_csv(csv_name, data_list):
    with open(csv_name,"w",encoding = "shift_jis",newline = "\n") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerows(data_list)

#nnabala_cli forward実行用関数
def nnabla_run(nnp_path :str, csv_path :str, output_path :str):
    a = 'nnabla_cli forward -c ' + nnp_path + ' -d ' + csv_path + ' -o ' + output_path
    subprocess.run(a,shell = True)

#推論実行結果の回収
def collect_parameter(inference_path :list, inference_file, output_path):
    #各推論結果の0.csvの読み取り
    df = [pd.read_csv(Path(i, inference_file), header=None) for i in inference_path]
    
    #convert nomalized parameter
    scale=[10,2,2,2,2,2,2,2,2]
    offset=[0,0,0,0,0,0,0,0,0]

    # df is not pandas DataFrame.df[j] is pandas DataFrame
    for j in range(len(df)):
        for i in range(9):
            df[j].loc[i]=df[j].loc[i]*scale[i]+offset[i]

    #各推論結果csvファイルの結合
    all_df = df[0]
    for i in range(len(df) - 1):
        all_df = pd.concat([all_df, df[i + 1]], axis=1)    #横方向に結合

    list = all_df.values.tolist()   #リストに変換

    output = [[Path(Path(i).parts[-2], Path(Path(i).parts[-1])) for i in inference_path]] #各条件のデレクトリ名作成
    output += list
    #csv書出し
    write_csv(output_path, output)
    return

def main():
    #nnpファイルのパス
    nnp_path = [
                r"D:\LS-DYNA_data\nnc\test004\NN_works3\nnc_files\hourglass_test004_205_001_3input_xstrain_to_alpha.files\20220918_221150\results.nnp",
                r"D:\LS-DYNA_data\nnc\test004\NN_works3\nnc_files\hourglass_test004_205_002_3input_thickness_to_alpha.files\20220918_221321\results.nnp",
                r"D:\LS-DYNA_data\nnc\test004\NN_works3\nnc_files\hourglass_test004_205_010_6input_x_y_strain.files\20220918_221441\results.nnp"
                ]
    #入力データのパスの入ったcsv
    csv_path = r"D:\LS-DYNA_data\nnc\test004\experiment\ex_path2.csv"
    # csv_path = r"D:\LS-DYNA_data\nnc\test004\experiment\nnabla_test_dataset.csv"
   
    #推論結果を保存するディレクトリのパス, 事前に作っておく必要がある
    inference_path = [
                      r"D:\LS-DYNA_data\nnc\test004\experiment\inference_data\205_001",
                      r"D:\LS-DYNA_data\nnc\test004\experiment\inference_data\205_002",
                      r"D:\LS-DYNA_data\nnc\test004\experiment\inference_data\205_010"
                      ]

    #引数の結合
    arg_list = []
    for i, j in zip(nnp_path, inference_path):
        arg_list.append([i, csv_path, j])
    #推論実行
    b = [nnabla_run(*i) for i in arg_list]
    #推論結果の回収・結合・書出し
    # inference_file_path = Path(r'0_0000\0.csv')
    # output_file_path = Path(r'D:\LS-DYNA_data\nnc\test004\experiment\all_inference_data.csv')
    # c = collect_parameter(inference_path, inference_file_path, output_file_path)

#以下実行部
if __name__ == '__main__':
    main()
