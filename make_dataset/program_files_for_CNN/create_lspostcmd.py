import numpy as np

def create_lspostcmd(cmdfile_name="lspostcmd_multi-section_cut.cfile"):
    center_point_coord=[0,1.5] # coordinate of the center point of specimen
    evaluation_area=[8.5,2]      # coordinate of the corner point of evaluation area
    evaluation_point_numbers=[35,11]    # dimension of 2d data

    # counter plot numbers of lsprepost and their name
    contour_value_list=[246,57,58] 
    contour_value_name_list=["x-coordinate","X_strain","Y_strain"]

    interval_x=evaluation_area[0]*2/(evaluation_point_numbers[0]-1)
    interval_y=evaluation_area[1]*2/(evaluation_point_numbers[1]-1)
    x_coord_of_points=np.arange(-evaluation_area[0],evaluation_area[0]+interval_x,interval_x)
    y_coord_of_points=np.arange(-evaluation_area[1],evaluation_area[1]+interval_y,interval_y)

    datalist=[]
    datalist.append(r'openc d3plot ".\d3plot"')
    datalist.append(r'ac')    
    datalist.append(r'state -1;')    
    datalist.append(r'range avgfrng node')    
    datalist.append(r'$')    

    for i in range(evaluation_point_numbers[1]): # point number along y direction
        n=0
        for j in contour_value_list:
            datalist.append(r'fringe {0}'.format(str(j)))
            datalist.append(r'pfringe')
            datalist.append(r'splane linewidth 1')
            datalist.append(r'splane fixsp')
            datalist.append(r'splane dep1 1  0.00  1.00  0.00')
            datalist.append(r'splane setbasept 0 {0} 10'.format(str(center_point_coord[1]+y_coord_of_points[i])))
            datalist.append(r'splane drawcut')
            datalist.append(r'splane plotcut 0 0')
            # datalist.append(r'xyplot 1 savefile ms_csv ".\{0}.csv" 1 all'.format(contour_value_name_list[n]+"_"+str("%.2f" % y_coord_of_points[i])))
            datalist.append(r'xyplot 1 savefile ms_csv ".\{0}.csv" 1 all'.format(contour_value_name_list[n]+"_"+str(i)))
            datalist.append(r'deletewin 1')
            datalist.append(r'splane done')
            datalist.append(r'$')
            n+=1    

    datalist="\n".join(datalist)
    with open(cmdfile_name, 'w') as f:
        f.writelines(datalist)


#実行部
if __name__ == '__main__':

    cmdfile_name="lspostcmd_multi-section_cut.cfile" # counter plot number of lsprepost

    # create lsprepost cmd file
    create_lspostcmd(cmdfile_name)        # Input a name of a created file