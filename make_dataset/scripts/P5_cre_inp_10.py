import csv
import os
import pprint
import copy
import numpy as np
from tqdm import tqdm

def P5(ref_y_disp_node, ref_y_disp_node2, model_scale):

    #csv書き込み関数
    def write_csv(csv_name, data_list):
        with open(csv_name,"w",encoding = "shift_jis",newline = "\n") as f:
            writer = csv.writer(f,lineterminator="\n")
            writer.writerows(data_list)
        return

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
        return parameter_list_for_keywrod

    #keywordファイルの番号を取得
    def cre_key_No():
        parameter_list_for_keywrod = read_parameter("parameter_success.csv")
        key_No = [i[0] for i in parameter_list_for_keywrod]

        return key_No

    #各keywordファイルのファイル名とパスを取得
    def cre_key_list():
        key_list = []
        key_No = cre_key_No()
        for i in key_No:
            # for j in ["_0", "_45", "_90"]:
            # for j in ["_0", "_90"]:
            for j in ["_0"]:
            

                a = ["keyword" + i + j + ".k", os.getcwd() + "/keyword/keyword" + i + "/keyword" + i + j]
                key_list.append(a)

        return key_list

    #Y_disp - Y_force カーブの作成
    def create_ydisp_yforce_curve(key_path, model_scale, ref_y_disp_node, ref_y_disp_node2):
        #Yforceの取り出し
        with open(key_path[1] + "/bndout","r") as f :
            list_f = f.readlines()
        
        list_f = [i for i in list_f if "yforce" in i]
        list_f = [float(i[47:57]) * model_scale for i in list_f]  # model_scale=1 in full model or 2 in quater model
        list_f_dataset = [i / 10000 for i in list_f]


        #y-dispの取り出し
        with open(key_path[1] + "/nodout","r") as f2 :
            list_disp = f2.readlines()

        list_y = [i for i in list_disp if ref_y_disp_node in i]
        list_y = list_y[1::2]
        list_y = [float(i[23:34]) * model_scale for i in list_y] #ydisp  # model_scale=1 in full model or 2 in quater model
        # list_y_dataset = [float(i) / 10 for i in list_y]        #ydispを1/10スケール

        list_y2 = [i for i in list_disp if ref_y_disp_node2 in i]
        list_y2 = list_y2[1::2]
        list_y2 = [float(i[23:34]) * model_scale for i in list_y2] #ydisp2  # model_scale=1 in full model or 2 in quater model
        list_y_dataset = [(2*float(i) - float(j)) / 10 for i,j in zip(list_y, list_y2)]        #ydisp2を1/10スケール

        #yforce,y-cordinateを1つのリストにまとめる
        ad_list_fy = []
        ad_list_fy_dataset = []
        for j in range(len(list_f)):
            a = [list_y[j], list_f[j]]
            ad_list_fy.append(a)
            b = [list_y_dataset[j], list_f_dataset[j]]
            ad_list_fy_dataset.append(b)

        return ad_list_fy, ad_list_fy_dataset
          
    #Y_disp - Y_force カーブのデータセットの作成    
    # def create_FS_dataset(key_path_list):

    #     print("creating FS dataset")
            
    #     list_fy = []
    #     list_fy_dataset = []

    #     # for i in key_path_list:
            
    #     #     #Y_disp - Y_force カーブの作成
    #     #     ad_list_fy, ad_list_fy_dataset = create_ydisp_yforce_curve(i, model_scale, ref_y_disp_node, ref_y_disp_node2)
    #     #     list_fy.append(ad_list_fy)   # 変位ー荷重曲線の2列データ
    #     #     list_fy_dataset.append(ad_list_fy_dataset)   # 変位ー荷重曲線を10000で割っての2列データ

    #     #keywordファイルの名前一覧の取得
    #     inp_path_list = []
    #     dataset_inp_path_list = []
    #     inp_csv_list = []
    #     dataset_inp_csv_list = []
    
    #     for i in cre_key_No():
    #         a = os.getcwd() + "/inp/inp" + i
    #         b = os.getcwd() + "/dataset_inp/dataset_inp" + i
    #         inp_path_list.append(a)
    #         dataset_inp_path_list.append(b)

    #         # for j in ["_0", "_45", "_90"]:
    #         for j in ["_0", "_90"]:

    #             a = os.getcwd() + "/inp/inp" + i
    #             b = os.getcwd() + "/dataset_inp/dataset_inp" + i
    #             c = a + '/inp_1_' + i + j + '.csv'
    #             d = b + '/dataset_inp_1_' + i + j + '.csv'
    


    #             inp_csv_list.append(c)
    #             dataset_inp_csv_list.append(d)


    #     for i, j in zip(inp_path_list, dataset_inp_path_list):
    #         os.makedirs(i, exist_ok = True)
    #         os.makedirs(j, exist_ok = True)

    #     pprint.pprint(inp_path_list)

    #     # for i, j, k, l in zip(inp_csv_list, dataset_inp_csv_list, list_fy, list_fy_dataset): # 保存するファイル名のリスト（＋正規化），データ
    #     #     write_csv(i, k)
    #     #     write_csv(j, l)
    #     return

    
    def create_FS_dataset(key_path_list):
        print("creating dataset path..")
    
        # Create path lists
        inp_path_list = [f"{os.getcwd()}/inp/inp{i}" for i in cre_key_No()]
        dataset_inp_path_list = [f"{os.getcwd()}/dataset_inp/dataset_inp{i}" for i in cre_key_No()]
        # csv_suffix = ["_0", "_90"]
        # inp_csv_list = [f"{path}/inp_1_{i}{suffix}.csv" for path, i, suffix in zip(inp_path_list, cre_key_No(), csv_suffix * len(cre_key_No()))]
        # dataset_inp_csv_list = [f"{path}/dataset_inp_1_{i}{suffix}.csv" for path, i, suffix in zip(dataset_inp_path_list, cre_key_No(), csv_suffix * len(cre_key_No()))]
    
        # Create directories
        for i, j in zip(inp_path_list, dataset_inp_path_list):
            os.makedirs(i, exist_ok=True)
            os.makedirs(j, exist_ok=True)
    
        pprint.pprint(inp_path_list)
    



    #read data as np.array
    def read_data_from_csv(file_name,header=2):
        csv_data = np.genfromtxt(file_name, delimiter=',',skip_header=header)

        return csv_data



    # 全てのkeywordフォルダにおいて，全てのひずみ（contour_value_name_list）を補間2Dにして，そのフォルダにcsv_file_nameのファイル名でcsv保存する
    def create_2d_csv_all(key_list,csv_file_names_list, x_coordinate_name_list, contour_value_name_list):

        total = len(key_list) * len(contour_value_name_list)
        with tqdm(total=total) as pbar:
            for i in key_list:
                for j in range(len(contour_value_name_list)):
                    csv_file_name=i[1] + '\\' + csv_file_names_list[j]
                    create_2d_csv(i[1],csv_file_name, x_coordinate_name_list[0], contour_value_name_list[j])
        pbar.update(1)
        return    

    # １組のx_coordinate, contour_valueを結合し，2D行列にして，csv_file_nameに保尊
    def create_2d_csv(key_path,csv_file_name, x_coordinate_name, contour_value_name):    
        # print("creating 2d csv")

        center_point_coord=[0,1.5] # coordinate of the center point of specimen
        evaluation_area=[6.5,1.5]      # coordinate of the corner point of evaluation area
        evaluation_point_numbers=[55,11]    # dimension of 2d data
        interval_x=evaluation_area[0]*2/(evaluation_point_numbers[0]-1)  # x-cord of interpolation
        interval_y=evaluation_area[1]*2/(evaluation_point_numbers[1]-1)

        x_coord_of_points=np.arange(-evaluation_area[0],evaluation_area[0]+interval_x,interval_x)
        y_coord_of_points=np.arange(-evaluation_area[1],evaluation_area[1]+interval_y,interval_y)
        with open("coordinate_list.csv","w",encoding = "shift_jis",newline = "\n") as f:
            writer = csv.writer(f,lineterminator="\n")
            writer.writerow(x_coord_of_points)
            writer.writerow(y_coord_of_points)

        ref_x=x_coord_of_points

        # interpolation to make dataset
        csv_data_2d=np.zeros((len(ref_x),evaluation_point_numbers[1]))


        for j in range(evaluation_point_numbers[1]):

            file_x_coord = key_path + "\{0}.csv".format(x_coordinate_name+"_"+str(j))
            file = key_path + "\{0}.csv".format(contour_value_name+"_"+str(j))

            # get calculation data
            cal_Xdata = read_data_from_csv(file_x_coord,2) # header=2
            cal_data = read_data_from_csv(file,2)

            if cal_Xdata[0,1] < 0: # ls-prepostのsection plotの開始がどちらのエッジが分からないため
                csv_data_2d[:,j]= np.interp(ref_x, cal_Xdata[:,1], cal_data[:,1])
            else:
                csv_data_2d[:,j]= np.interp(ref_x, cal_Xdata[::-1][:,1], cal_data[::-1][:,1]) #並び順を昇順に反転

        write_csv(csv_file_name,csv_data_2d)

        return

    def create_2d_dataset(key_path_list,csv_file_name_list,data_type_list,strain_scale):

        inp_path_list = []
        dataset_inp_path_list = []
        inp_csv_list = []
        dataset_inp_csv_list = []
    
        for i in cre_key_No():
            a = os.getcwd() + "/inp/inp" + i
            b = os.getcwd() + "/dataset_inp/dataset_inp" + i
            inp_path_list.append(a)
            dataset_inp_path_list.append(b)

            # for j in ["_0", "_45", "_90"]:
            # for j in ["_0", "_90"]:
            for j in ["_0"]:
            

                a = os.getcwd() + "/inp/inp" + i
                b = os.getcwd() + "/dataset_inp/dataset_inp" + i
                for k in data_type_list[1:]: # data_type_listの最初の番号1はFS用なので外し，2番目から

                    c = a + '/inp_{0}_{1}{2}.csv'.format(k, i, j)   # 2Dデータの番号はls-prepostと一致．57->x strain, 58->y strain
                    d = b + '/dataset_inp_{0}_{1}{2}.csv'.format(k, i, j) 
                    inp_csv_list.append(c)
                    dataset_inp_csv_list.append(d)


        for i, j in zip(inp_path_list, dataset_inp_path_list):
            os.makedirs(i, exist_ok = True)
            os.makedirs(j, exist_ok = True)

        pprint.pprint(inp_path_list)


        for i in tqdm(range(len(key_path_list)), desc='Processing folders (create_2d_dataset)'): 
        # for i in range(len(key_path_list)): # 保存するファイル名のリスト，データ
            for j in range(len(csv_file_name_list)):
                file_name=key_path_list[i][1] + '\\' + csv_file_name_list[j]
                data=read_data_from_csv(file_name,header=0)
                dataset=data*strain_scale # strain_scale はひずみの正規化の倍率
                write_csv(inp_csv_list[len(csv_file_name_list)*i+j], data)
                write_csv(dataset_inp_csv_list[len(csv_file_name_list)*i+j], dataset)

        # for j, l in zip(dataset_inp_csv_list, list_xy_dataset): # 保存するdatasetファイル名のリスト，データ
        #     write_csv(j, l)

        return


    #X_cordinate - y-value カーブの作成
    def create_xcord_yvalue_curve(key_path, filename, ref_x, scale_x, scale_y):
        #x-cordinateとy-valueの取り出し
        with open(key_path[1] + filename,"r") as f2 :
            list_trd = f2.readlines()

        Xdata =np.array([float(i.split(',')[0]) for i in list_trd[2::1]]) 
        Xdata = Xdata - np.max(Xdata)/2  #Full model 用の原点位置の変更
        Ydata =np.array([float(i.split(',')[1]) for i in list_trd[2::1]])
        list_x = ref_x
        list_y = np.interp(ref_x, Xdata, Ydata)      #補間されたYdata

        list_x_dataset = [float(i) / scale_x for i in list_x]        #x-cordinateをscale_xスケール
        list_y_dataset = [float(i) / scale_y for i in list_y]        #y-valueをscale_yスケール

        #x-cordinate,y-valueを1つのリストにまとめる
        ad_list_xy1 = []
        ad_list_xy1_dataset = []
        for j in range(len(list_y)):
            a = [list_x[j], list_y[j]]
            ad_list_xy1.append(a)
            b = [list_x_dataset[j], list_y_dataset[j]]
            ad_list_xy1_dataset.append(b)
 
        return ad_list_xy1, ad_list_xy1_dataset

    # 実行部
    csv_file_name_list=["X_strain_2D_55-11.csv","Y_strain_2D_55-11.csv"] # counter plot number of lsprepost
    x_coordinate_name_list=["x-coordinate"]
    contour_value_name_list=["X_strain","Y_strain"]
    data_type_list=[1,57,58]

    key_path_list = cre_key_list()  # 0,45,90を含むkeywordファイルのファイル名と全フォルダリストを取得
    create_FS_dataset(key_path_list)
    print("------Finish create_FS dataset------")

    create_2d_csv_all(key_path_list,csv_file_name_list, x_coordinate_name_list, contour_value_name_list)
    print("------Finish create_2d_csv_all------")
    strain_scale=5
    create_2d_dataset(key_path_list,csv_file_name_list,data_type_list,strain_scale)
    print("------Finish create_2d_dataset------")



    # create dataset_inp_path.csv
    # csv_path = [['x1:FS_0deg', 'x2:FS_45deg', 'x3:FS_90deg', 'x4__0:X2D_0deg', 'x5__0:X2D_45deg', 'x6__0:X2D_90deg', 
    #              'x4__1:Y2D_0deg', 'x5__1:Y2D_45deg', 'x6__1:Y2D_90deg']]
    csv_path = [['x1:FS_0deg', 'x3:FS_90deg', 'x4__0:X2D_0deg',  
                 'x4__1:Y2D_0deg']]


    # RD_list=['_0.csv','_45.csv','_90.csv']
    # RD_list=['_0.csv','_90.csv']
    RD_list=['_0.csv']



    for i in cre_key_No():
        ad_list=[]
        for j in data_type_list:
            for k in RD_list:
                a = os.getcwd() + "/dataset_inp/dataset_inp" + i + '/dataset_inp_{0}_'.format(j) + i + k
                ad_list.append(a)

        csv_path.append(ad_list)
    
    print("Writing csvs......")
    write_csv('dataset_inp_path.csv', csv_path)

    os.rename('./dataset_inp_path.csv', './dataset_inp/dataset_inp_path.csv')
    os.rename('./coordinate_list.csv', './dataset_inp/coordinate_list.csv')

    return

def P5_2():

    print("Running P5_2......")
    #csvファイルの値をリスト（ndarrayではない）に読み取り
    def read_csv(csv_name):
        with open(csv_name,'r') as paraf :
            para_read = csv.reader(paraf)
            parameter_list = [a for a in para_read]
        return parameter_list

    #csvファイルにデータリストを書き込み        
    def write_csv(csv_name, data_list):
        with open(csv_name,"w",encoding = "shift_jis",newline = "\n") as f:
            writer = csv.writer(f,lineterminator="\n")
            writer.writerows(data_list)
        return

    data_type_list=[1,57,58]
    # inputはsymmetyの操作をしていない，outputは標準偏差を先に求めるためsymmetyの操作ずみ
    dataset_input=read_csv(r"./dataset_inp/dataset_inp_path.csv",)  
    # dataset_output=read_csv(r"D:\LS-DYNA_data\nnc\test005\execution_test\dataset_outp\dataset_outp.csv",)
    dataset_output=read_csv(r"./dataset_outp/dataset_outp.csv",)


    dataset=[]  #datasetとして，input,outputを結合
    for i in tqdm(range(len(dataset_input)), desc='Processing data (input+output)'):
        dataset.append(dataset_input[i]+dataset_output[i])

    # symmetric operation
    symmetry_dataset_input=copy.deepcopy(dataset_input)[1:] # deepcopyでリストの参照を切って初期値代入
    symmetry_dataset_output=dataset_output[len(dataset_input):] # outputは後半がsymmetyの操作済みなので読み込むだけ

    for i in range(len(symmetry_dataset_input)):
        for j in range(len(data_type_list)):
            # symmetry_dataset_input[i][3*j],symmetry_dataset_input[i][3*j+2]=symmetry_dataset_input[i][3*j+2],symmetry_dataset_input[i][3*j]
            symmetry_dataset_input[i][2*j],symmetry_dataset_input[i][2*j+1]=symmetry_dataset_input[i][2*j+1],symmetry_dataset_input[i][2*j]
            
    # print(symmetry_dataset_input)
    for i in tqdm(range(len(symmetry_dataset_input)), desc='Processing symmetry data (input+output)'):
    # for i in range(len(symmetry_dataset_input)):
        dataset.append(symmetry_dataset_input[i]+symmetry_dataset_output[i])  
    
    write_csv("./dataset.csv",dataset)
    print("finish")

    return

#以下実行部
if __name__ == '__main__':
    P5()
