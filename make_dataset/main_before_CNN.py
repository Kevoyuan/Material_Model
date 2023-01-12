from program_files_for_CNN import P1_cre_para_csv_3 as P1
from program_files_for_CNN import P2_cre_key_files as P2
from program_files_for_CNN import P3_cre_outp_standardization2 as P3
from program_files_for_CNN import P4_dyna_lsprepost_run as P4
from program_files_for_CNN import P4_other 

from program_files_for_CNN import P5_cre_inp_10 as P5
from program_files_for_CNN import standardization_symm as std_symm

import os
import numpy as np

from program_files_for_CNN import split_data as split

if __name__ == '__main__':
    sig45_range = [0.9, 1.2]
    sig90_range = [0.9, 1.2]
    sigb_range = [0.9, 1.25]
    r00_range = [0.9, 3.0]
    r45_range = [0.9, 3.0]
    r90_range = [0.9, 3.0]
    rb_range = [0.7, 1.3]
    M_range = [5,10]   #　Mは実数でもよい

    # 実行フォルダの作成と変更 training_dataは'./execution'，evaluation_dataは'./execution2'
    # execution_Path='./execution'
    execution_Path='./roughmodel_1024'
    os.makedirs(execution_Path,exist_ok=True)
    os.chdir(execution_Path)

    # creation of  random sigma, r-value and M. and identification of alpha1-8 of yld2000-2d
    #(number of parameter sets, range of parameters:sigma,r,M)
    # P1.P1(1024, sig45_range, sig90_range, sigb_range, r00_range, r45_range, r90_range, rb_range, M_range)
    # print('-----Finish P1------')

    # creation of ls-dyna keyword file
    # P2.P2()
    # print('-----Finish P2------')

    # standardization of M and alpha1-8 for dataset_output
    # P3.P3()
    # print('-----Finish P3------')

    # Run ls-dyna pararelly and run ls-prepost
    #使用するcpu数, LS-DYNAの解析ソルバーパス，LS-Prepostのパス，.cfileのパス
    cpus = 20
    # solver_path = "D:\LSDYNA\program\ls-dyna_smp_d_R11_1_0_winx64_ifort160.exe" 
    # solver_path = "C:\LSDYNA\LSDYNA\program\ls-dyna_smp_d_R12_0_0_winx64_ifort170.exe"
    solver_path = "C:\LSDYNA\LS-PrePost\ls-dyna_smp_d_R12_0_0_winx64_ifort170\ls-dyna_smp_d_R12_0_0_winx64_ifort170.exe"
    # prepost_path = 'D:\LSDYNA\program\lsprepost.exe'
    # prepost_path = "C:\LSDYNA\LSDYNA\program\lsprepost4.8_x64.exe"
    prepost_path = "C:\LSDYNA\LS-PrePost\lsprepost4.8_x64.exe"
    # cfile_path='..\..\..\..\optional_files\lspostcmd_multi-section_cut_35-11.cfile'
    cfile_path='..\..\..\..\optional_files\lspostcmd_multi-section_cut.cfile'



    # P4.P4(cpus, solver_path)
    # P4_other.P4(cpus, solver_path)
    # P4.P4_2(cpus, prepost_path, cfile_path)
    # print('-----Finish P4------')

    # reference position of x
    # ref_x = np.arange(-8.5,9,0.5)
    # ref_y_disp_node = "      330"
    # ref_y_disp_node2 = "      326"
    ref_y_disp_node = "      247" 
    ref_y_disp_node2 = "      566"
    model_scale = 1 # full model:1, quater model:2

    # P5.P5(ref_y_disp_node, ref_y_disp_node2, model_scale)
    # P5.P5_2()
    # print('-----Finish P5------')   

    # std_symm.standardization_symm(execution_Path)

    # split.split_data(execution_Path)
    # print('-----Finish Split Data------')   
