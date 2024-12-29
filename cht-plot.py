import math
import os,sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir(sys.path[0])

def export_TEM(file_name):
    print('------------------------')

    bi_data = pd.read_csv('Bi_positions.txt', sep='\s+', header=None)
    fe_data = pd.read_csv('Fe_positions.txt', sep='\s+', header=None)

    # Assigning column names for clarity
    bi_data.columns = ['x', 'y']
    fe_data.columns = ['x', 'y']

    bi_data = bi_data[bi_data['x'] > fe_data['x'].min()]
    bi_data = bi_data[bi_data['x'] < fe_data['x'].max()]
    bi_data = bi_data[bi_data['y'] > fe_data['y'].min()]
    bi_data = bi_data[bi_data['y'] < fe_data['y'].max()]

    fe_data = fe_data[fe_data['x'] > bi_data['x'].min()]
    fe_data = fe_data[fe_data['x'] < bi_data['x'].max()]
    fe_data = fe_data[fe_data['y'] > bi_data['y'].min()]
    fe_data = fe_data[fe_data['y'] < bi_data['y'].max()]

    fe_data = fe_data.reset_index(drop=True)
    bi_data = bi_data.reset_index(drop=True)

    nn = fe_data.shape[0]
    mm = bi_data.shape[0]
    FeBi = [[0 for j in range(4)] for i in range(nn)]


    flag = 1
    for i in range(nn):
        count=0
        for j in range(mm):
            x = abs(fe_data['x'][i] - bi_data['x'][j])
            y = abs(fe_data['y'][i] - bi_data['y'][j])
            if x < 70.0 and y < 70.0:
                count+=1
                FeBi[i][0] += fe_data['x'][i] - bi_data['x'][j]
                FeBi[i][1] += fe_data['y'][i] - bi_data['y'][j]

        if count < 4:
            print('distance set too small')
            FeBi[i][0] = 0
            FeBi[i][1] = 0
            flag = 0
        elif count > 4:
            print('distance set too large')
            FeBi[i][0] = 0
            FeBi[i][1] = 0
            flag = 0
    if flag:
        print('every Fe atoms find 4 adjacent Bi atoms')

    with open('result.txt', "w") as out_file:
        x_list = []
        y_list = []
        u_list = []
        v_list = []
        angle_list = []
        for i in range(nn):
            Positionx = fe_data['x'][i]
            Positiony = fe_data['y'][i]
            changdu = math.sqrt(FeBi[i][0]**2 + FeBi[i][1]**2)/4
            jiaodu = math.atan2(FeBi[i][1], FeBi[i][0])
            Arrowx = FeBi[i][0]/4
            Arrowy = FeBi[i][1]/4
            Unitx = FeBi[i][2]/2
            Unity = FeBi[i][3]/2
            out_file.write(f"{Positionx:.6f} {Positiony:.6f} {jiaodu:.6f} {changdu:.6f} {Arrowx:.6f} {Arrowy:.6f} {Unitx:.6f} {Unity:.6f}\n")
            x_list.append(Positionx)
            y_list.append(Positiony)
            u_list.append(Arrowx)
            v_list.append(Arrowy)
            angle_list.append(jiaodu)

        # x_list,y_list = np.meshgrid(x_list,y_list)
        # u_list,v_list = np.meshgrid(u_list,v_list)

        fig = plt.figure(figsize=(5,4))
        plt.subplots_adjust(left=0.2,right=0.9,top=0.9,bottom=0.2)
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        ax.quiver(x_list,y_list,u_list,v_list,angle_list)
        # print('------------------------')
        # print(file_name)
        # print('all_x:',sum(u_list))
        # print('all_y:',sum(v_list))
        # print('------------------------')
        plt.savefig('result.png',dpi=300)


if __name__ == '__main__':
    export_TEM(1)