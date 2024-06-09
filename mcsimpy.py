import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from interactions import interactions
# from endmovetest import endmovetest
# from cornermovetest import cornermovetesmct
# from params import params
import math
import random
import time
import csv

st_time = time.time()
# Rg and end-to-end distance calculator
def params(p):
    com = np.zeros(2)
    com[0] = np.mean(p['x'])
    com[1] = np.mean(p['y'])
    s_sq = (p['x'] - com[0])**2 + (p['y'] - com[1])**2
    gr = 1/16 * np.mean(np.sum(s_sq))
    ee_dist = np.sqrt((p['x'][0] - p['x'][15])**2 + (p['y'][0] - p['y'][15])**2)
    return gr, ee_dist

# Corner move function
# works by checking for L-shaped residue corners and checking for possible 
# corner moves
def cornermovetest(p, r):
    p1 = p.copy()
    dist = math.sqrt((p['x'][r-1] - p['x'][r+1])**2 + (p['y'][r-1] - p['y'][r+1])**2)
# temporary positions
    if dist < 1.5:
        t1 = [p['x'][r+1], p['y'][r-1]]
        t2 = [p['x'][r-1], p['y'][r+1]]

        f2 = 1
        f1 = 1
# flag for available positions
        for i in range(16):
            if p['x'][i] == t2[0] and p['y'][i] == t2[1]:
                f2 = 0
            elif p['x'][i] == t1[0] and p['y'][i] == t1[1]:
                f1 = 0

        if f1==0 and f2 == 0:
            return p1
# flag for positions which have residues
        if p['x'][r] == t1[0] and p['y'][r] == t1[1]:
            f1 = 2
        elif p['x'][r] == t2[0] and p['y'][r] == t2[1]:
            f2 = 2
# update the positions 
        if f1 == 2 and f2 == 1:
            p['x'][r] = t2[0]
            p['y'][r] = t2[1]
        elif f2 == 2 and f1 == 1:
            p['x'][r] = t1[0]
            p['y'][r] = t1[1]

# corner move metropolis criterion
    en = interactions(p, r)
    e_del = interactions(position, r) - interactions(p, r)

    w = np.exp(-e_del)
    if w > 1:
        return p
    else:
        random_c = random.random()
        if w > random_c:
            return p
        else:
            return p1
        
# end move function
# this works by pivoting based on availability of the positions around
# the next residue to the given residue
def endmovetest(p, r):
    p1 = p.copy()
    if r == 15:
        # define temp positions
        if p.x[r-1] == p.x[r]:
            t = [
                [p.x[r-1]-1, p.y[r-1]],
                [p.x[r-1]+1, p.y[r-1]]
            ]

        else:
            t = [
                    [p.x[r-1], p.y[r-1]+1],
                    [p.x[r-1], p.y[r-1]-1]
                ]
    else:
        if p.x[r+1] == p.x[r]:
           t = [
               [p.x[r+1]-1, p.y[r+1]],
               [p.x[r+1]+1, p.y[r+1]]
           ] 
        else:
            t = [
                [p.x[r+1], p.y[r+1]+1],
                [p.x[r+1], p.y[r+1]-1]
            ]
        
        flag = [1, 1]

        # flag is positions available
        for i in range(15):
            if p.x[i] == t[0][0] and p.y[i] == t[0][1]:
                flag[0] = 0
            elif p.x[i] == t[1][0] and p.y[i] == t[1][1]:
                flag[1] = 0
        
        if flag[0] and flag[1]:
            rnd = random.randint(0,1)
            p.x[r] = t[rnd][0]
            p.y[r] = t[rnd][1]
        elif flag[0]:
            p.x[r] = t[0][0]
            p.y[r] = t[0][1]
        elif flag[1]:
            p.x[r] = t[1][0]
            p.x[r] = t[1][1]
    
    # Metropolis criterion
    en = interactions(p, intPB)
    e_del = interactions(p1, intPB) - en
    
    w = np.exp(e_del)
   
    if w > 1:
        return p
    else:
        random_c = random.random()
        if w > random_c:
            return p
        else:
            return p1

# interaction energies function
def interactions(pos, intEnergy):
    eMat = [[0, 0] for i in range(9)]
    distance = lambda x1, y1, x2, y2: math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    totalE = 0
    eMat[0][1] = distance(pos['x'][0], pos['y'][0], pos['x'][11], pos['y'][11])
    eMat[1][1] = distance(pos['x'][0], pos['y'][0], pos['x'][3], pos['y'][3])
    eMat[2][1] = distance(pos['x'][2], pos['y'][2], pos['x'][5], pos['y'][5])
    eMat[3][1] = distance(pos['x'][3], pos['y'][3], pos['x'][10], pos['y'][10])
    eMat[4][1] = distance(pos['x'][4], pos['y'][4], pos['x'][9], pos['y'][9])
    eMat[5][1] = distance(pos['x'][4], pos['y'][4], pos['x'][7], pos['y'][7])
    eMat[6][1] = distance(pos['x'][8], pos['y'][8], pos['x'][15], pos['y'][15])
    eMat[7][1] = distance(pos['x'][9], pos['y'][9], pos['x'][14], pos['y'][14])
    eMat[8][1] = distance(pos['x'][10], pos['y'][10], pos['x'][13], pos['y'][13])

    for i in range(9):
        if eMat[i][1] == 1:
            eMat[i][0] = intEnergy
        else:
            eMat[i][0] = 0
    # loop for random interactions 
    # for p in range(16):
    #     for q in range(16):
    #         if abs(p-q)>1:
    #             temp_dis = distance(pos.x[p], pos.y[p], pos.x[q], pos.y[q])
    #             if temp_dis == 1:
    #                 totalE = totalE + intEnergy

    totalE = sum([row[0] for row in eMat])
    return totalE

intPBvals = np.array([-0.25, -0.75, -1.25, -2.5])
cycles = 10000
intPB = intPBvals[0]

reps = 1
output_data = np.zeros((3, cycles, reps, 4))

# output data is a 4D matrix dataset. Composed of 4 3D datasets. 
# Each 3D dataset has indices 
# 1 --> Energy
# 2 --> Radius of gyration
# 3 --> End to end distance
# there are 50 repetitions of simulations for each value of interaction
# energy per non covalent interaction. 
# The 4th dimension is the various
# energy values as indexed in intPBvals. 

# output_data_1 = pd.DataFrame({'Energy': [], 'R_gyr': [], 'EE_dist': []})

position = pd.read_excel("native state values.xlsx")

plt_var, = plt.plot(position.x, position.y, "-o", color=[0, 0, 0], linewidth=1.5)

plt.grid(True)
plt.xlim(-7, 11)
plt.ylim(-7, 11)

for repcount in range(reps):
    position = pd.read_excel("native state values.xlsx")
    
    # Main loop 
    for i in range(cycles):
        energy = interactions(position, intPB)
        # print('initial energy: ', energy)
        plt.title(f"energy = {energy}kT, cycles = {i}, rep count = {repcount}")
        plt.pause(0.001)
        plt_var.set_xdata(position.x)
        plt_var.set_ydata(position.y)

        res = np.random.randint(0, 15)
        # print('residue', res)
        
        # time.sleep(1)

        if res == 0 or res == 15:
            position = endmovetest(position, res)
        else:
            position = cornermovetest(position, res)
        
        # resetting the plot values
        plt_var.set_xdata(position.x)
        plt_var.set_ydata(position.y)
        plt.draw()
        
        # taking output data from the iterations
        gr, eed = params(position)
        output_data[0, i, repcount, 0] = energy
        output_data[1, i, repcount, 0] = gr
        output_data[2, i, repcount, 0] = eed
        if i % 1000 == 0:
            print(i)
    print(repcount)

energy_data = output_data[0, :, :, 0]
gr_data = output_data[1, :, :, 0]
eed_data = output_data[2, :, :, 0]

with open('output_data_125_rand_en.csv', mode='w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(energy_data)

with open('output_data_125_rand_gr.csv', mode='w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(gr_data)

with open('output_data_125_rand_eed.csv', mode='w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(eed_data)







        
       
        
