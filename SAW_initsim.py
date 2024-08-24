import numpy as np
#import scipy as sc
#import matplotlib.pyplot as plt
#from sympy.utilities.iterables import variations
import time
import itertools
#from matplotlib.ticker import PercentFormatter
#from scipy.optimize import curve_fit
import warnings
import time as t
warnings.filterwarnings("ignore")
#%matplotlib inline

# GLOBAL CONSTANTS
J=1
h=0
K=2

def create_initial(rows, cols):
    return np.random.choice((-1,1), size=(rows,cols))

def saw(n):
    x, y = [n], [n]
    positions = set([(n,n)])  #positions is a set that stores all sites visited by the walk
    for i in range(n-1):
        directions = [(1,0), (0,1), (-1,0), (0,-1)]
        directions_feasible = []  #directions_feasible stores the available directions
        for dx, dy in directions:
            if (x[-1] + dx, y[-1] + dy) not in positions:  #checks if direction leads to a site not visited before
                directions_feasible.append((dx,dy))
        if directions_feasible:  #checks if there is a direction available
            dx, dy = directions_feasible[np.random.randint(0,len(directions_feasible))]  #choose a direction at random among available ones
            positions.add((x[-1] + dx, y[-1] + dy))
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
        else:  #in that case the walk is stuck
            break  #terminate the walk prematurely
    return x, y

def free_list(positions_list, spin_array):
    for i in range(1,-1,-1):
        if not (spin_array[positions_list[i][0]][positions_list[i][1]] == 0):
            positions_list.remove(positions_list[i])
    return positions_list

def endpoints(x,y,i,spin_array):
    if i==0:
        if x[1] == x[0]-1 and y[1]==y[0]:               #1
            pos_pos =free_list([[x[i]-1,y[i]+1],[x[i]-1,y[i]-1]], spin_array)
            if len(pos_pos) != 0:
                j = np.random.randint(0,len(pos_pos))
                return pos_pos[j]
        if x[1] == x[0] and y[1]==y[0]-1:                #2
            pos_pos = free_list([[x[i]-1,y[i]-1],[x[i]+1,y[i]-1]], spin_array)
            if len(pos_pos) != 0:
                j = np.random.randint(0,len(pos_pos))
                return pos_pos[j]
        if x[1] == x[0]+1 and y[1]==y[0]:                #3
            pos_pos = free_list([[x[i]+1,y[i]+1],[x[i]+1,y[i]-1]],spin_array)
            if len(pos_pos) != 0:
                j = np.random.randint(0,len(pos_pos))
                return pos_pos[j]
        if x[1] == x[0] and y[1]==y[0]+1:                #4
            pos_pos = free_list([[x[i]-1,y[i]+1],[x[i]+1,y[i]+1]], spin_array)
            if len(pos_pos) != 0:
                j = np.random.randint(0,len(pos_pos))
                return pos_pos[j]
    if i==len(x)-1:
        if x[i]-1 == x[i-1] and y[i]==y[i-1]:               #1
            pos_pos = free_list([[x[i]-1,y[i]+1],[x[i]-1,y[i]-1]],spin_array)
            if len(pos_pos) != 0:
                j = np.random.randint(0,len(pos_pos))
                return pos_pos[j]
        if x[i] == x[i-1] and y[i]-1==y[i-1]:                #2
            pos_pos = free_list([[x[i]-1,y[i]-1],[x[i]+1,y[i]-1]], spin_array)
            if len(pos_pos) != 0:
                j = np.random.randint(0,len(pos_pos))
                return pos_pos[j]
        if x[i]+1 == x[i-1] and y[i]==y[i-1]:                #3
            pos_pos = free_list([[x[i]+1,y[i]+1],[x[i]+1,y[i]-1]], spin_array)
            if len(pos_pos) != 0:
                j = np.random.randint(0,len(pos_pos))
                return pos_pos[j]
        if x[i] == x[i-1] and y[i]+1==y[i-1]:                #4
            pos_pos = free_list([[x[i]-1,y[i]+1],[x[i]+1,y[i]+1]], spin_array)
            if len(pos_pos) != 0:
                j = np.random.randint(0,len(pos_pos))
                return pos_pos[j]
    return []

def correct_x(x, y, i, spin_array):
    possible = False
    positions = []
    if i == 0 or i == len(x) - 1:
        positions = endpoints(x, y, i, spin_array)
        if len(positions) != 0:
            possible = True
        return possible, positions
    if y[i] + 1 == y[i - 1] and x[i] == x[i - 1] and y[i + 1] == y[i] and x[i] + 1 == x[i + 1] and spin_array[x[i] + 1][
        y[i] + 1] == 0:  # 1
        positions = [x[i] + 1, y[i] + 1]
        possible = True

    if y[i] == y[i - 1] and x[i] - 1 == x[i - 1] and y[i + 1] == y[i] - 1 and x[i] == x[i + 1] and spin_array[x[i] - 1][
        y[i] - 1] == 0:  # 2
        positions = [x[i] - 1, y[i] - 1]
        possible = True

    if y[i] == y[i - 1] and x[i] + 1 == x[i - 1] and y[i + 1] == y[i] + 1 and x[i] == x[i + 1] and spin_array[x[i] + 1][
        y[i] + 1] == 0:  # 3
        positions = [x[i] + 1, y[i] + 1]
        possible = True

    if y[i] - 1 == y[i - 1] and x[i] == x[i - 1] and y[i + 1] == y[i] and x[i] - 1 == x[i + 1] and spin_array[x[i] - 1][
        y[i] - 1] == 0:  # 4
        positions = [x[i] - 1, y[i] - 1]
        possible = True

    if y[i] == y[i - 1] and x[i] + 1 == x[i - 1] and y[i + 1] == y[i] - 1 and x[i] == x[i + 1] and spin_array[x[i] + 1][
        y[i] - 1] == 0:  # 5
        positions = [x[i] + 1, y[i] - 1]
        possible = True
        return possible, positions
    if y[i] + 1 == y[i - 1] and x[i] == x[i - 1] and y[i + 1] == y[i] and x[i] - 1 == x[i + 1] and spin_array[x[i] - 1][
        y[i] + 1] == 0:  # 6
        positions = [x[i] - 1, y[i] + 1]
        possible = True

    if y[i] - 1 == y[i - 1] and x[i] == x[i - 1] and y[i + 1] == y[i] and x[i] + 1 == x[i + 1] and spin_array[x[i] + 1][
        y[i] - 1] == 0:  # 7
        positions = [x[i] + 1, y[i] - 1]
        possible = True

    if y[i] == y[i - 1] and x[i] - 1 == x[i - 1] and y[i + 1] == y[i] + 1 and x[i] == x[i + 1] and spin_array[x[i] - 1][
        y[i] + 1] == 0:  # 8
        positions = [x[i] - 1, y[i] + 1]
        possible = True
    return possible, positions

def neighbors(spin_array,N, x,y):
    left   = (x, (y-1))
    right  = (x, (y+1))
    top    = ((x-1), y)
    bottom = ((x+1), y)
    '''left   = (x, (y+1+N)%N)
    right  = (x, (y-1+N)%N)
    top    = ((x+1+N)%N, y)
    bottom = ((x-1+N)%N, y)'''
    return [spin_array[left[0], left[1]],
            spin_array[right[0], right[1]],
            spin_array[top[0], top[1]],
            spin_array[bottom[0], bottom[1]]]

def energy(spin_array, N, x_pos ,y_pos):
    return (-J*spin_array[x_pos][y_pos]*sum(neighbors(spin_array, N, x_pos, y_pos)) - h*spin_array[x_pos][y_pos])

def total_energy(spin_array,N,x,y):
    spin_list = merge_lists(x,y)
    # other (longer) way to calculate the energy used as a check 
    '''sum_energy = 0
    for i in range(len(spin_list)):
        print("position: ",spin_list[i][0],spin_list[i][1])
        print("neighbours: ", neighbors(spin_array, N, spin_list[i][0], spin_list[i][1]))
        print("energy:",energy(spin_array,N, spin_list[i][0], spin_list[i][1]))
    print(K*len(x))'''
    return (sum([energy(spin_array,N, spin_list[i][0], spin_list[i][1]) for i in range(len(spin_list))]) - K*len(x))

def merge_lists(x,y):
    return [[x[i],y[i]] for i in range(len(x))]

def spin_on_saw(spin_array,x,y):
    spins_list = merge_lists(x,y)
    for i in range(len(spin_array)):
        for j in range(len(spin_array[i])):
            if [i,j] not in spins_list:
                spin_array[i][j] = 0
    return spin_array

def create_high_initial(spin_array,x,y):
    spins_list = merge_lists(x,y)
    for i in range(len(spin_array)):
        for j in range(len(spin_array[i])):
            if [i,j] in spins_list:
                spin_array[i][j] = 1
    return spin_array

def plus_min(spin_array):
    plus_spin = []
    min_spin = []
    for i in range(len(spin_array)):
        for j in range(len(spin_array)):
            if spin_array[i][j] == 1:
                plus_spin.append([i,j])
            if spin_array[i][j] == -1:
                min_spin.append([i,j])
    return plus_spin, min_spin

def find_possible_points(x,y,spins):
    possible_points = []
    for i in range(0,len(x)):
        possible, positions = correct_x(x,y,i,spins)
        if possible:
            possible_points.append(i) #test: [x[i],y[i],i]
    if len(possible_points) ==0:
        raise ValueError('list of possible points must be non-empty, the SAW cannot move.')
    return possible_points

def region_counter(spin_array,x,y):
    counter = 1
    for i in range(len(x)-1):
        if not(spin_array[x[i]][y[i]] == spin_array[x[i+1]][y[i+1]]):
            counter += 1
    return counter

def spins_1dising(spin_array,x,y):
    spin_1d = [0]#[0]
    for i in range(len(x)):
        spin_1d.append(spin_array[x[i]][y[i]])
    spin_1d.append(0)
    return spin_1d

def energy_ising_1d(spin_list):
    sum_energy = 0
    for i in range(1,len(spin_list)-1):
        sum_energy += -J*spin_list[i]*(spin_list[i-1]+spin_list[i+1])
    sum_energy += -h*sum(spin_list)
    sum_energy += -K*(len(spin_list)-2)
    return sum_energy

def metropol_spin(spinsl,sweeps,T,xl,yl):
    β = 1/(T)
    mag = 0#np.zeros(sweeps)
    Energy = 0#np.zeros(sweeps)
    for sweep in range(sweeps):
        i = np.random.randint(0,len(xl))
        x_pos = xl[i] #np.random.randint(0,99)
        y_pos = yl[i] #np.random.randint(0,99)
        E_i = 2*energy(spins,n,x_pos,y_pos)
        ΔE = -E_i #E_f-E_i
        r = np.random.uniform()
        if ΔE <=0 or r<= np.exp(-β*ΔE):
            spinsl[x_pos][y_pos] = - spinsl[x_pos][y_pos]
        #Magnetization
        mag  = sum(sum(spins))/ (len(xl)) #[sweep]
        #Energy
        Energy = total_energy(spinsl,n,xl,yl) #[sweep]
        #print(mag[sweep])
    return [spinsl, mag, Energy,xl,yl]

def metropol_saw(spinsl,sweeps,T,xl,yl):
    β = 1/(T)
    mag = 0#np.zeros(sweeps)
    Energy = 0#np.zeros(sweeps)
    #activity = 0
    for sweep in range(sweeps):
        '''x2 = x.copy()
        y2 = y.copy()
        spins2 = spins.copy()'''
        i = np.random.choice(find_possible_points(xl,yl,spinsl))
        #print(find_possible_points(xl,yl,spinsl))
        #print(i)
        possp = find_possible_points(xl,yl,spinsl)
        x_pos = xl[i]
        y_pos = yl[i]
        activity = i
        possible, positions = correct_x(xl,yl,i,spinsl)
        if possible:
            E_i = energy(spinsl,n,x_pos,y_pos)
            #config_copy = metropolis_saw_move(i)
            spinsl,xl,yl = metropolis_saw_move(i,xl,yl,spinsl)
            #E_i = energy(config,n,x_pos,y_pos)
            x_pos = xl[i]
            y_pos = yl[i]
            E_f = energy(spinsl,n,xl[i],yl[i])
            ΔE = E_f-E_i
            r = np.random.uniform()
            if not(ΔE <=0 or r<= np.exp(-β*ΔE)):
                #print('accepted')
                spinsl,xl,yl = metropolis_saw_move(i,xl,yl,spinsl)
                activity = -1
                '''x = x2.copy()
                y = y2.copy()
                spins = spins2.copy()'''
            #else:
                #print('rejected')
        #print(activity)
        #print(possp)
        #Magnetization
        mag= sum(sum(spinsl))/(len(xl))#[sweep]
        #Energy
        Energy = total_energy(spinsl,n,xl,yl)#[sweep]
        #print(mag[sweep])
    return [spinsl, mag, Energy,xl,yl, activity,possp]

def metropolis_saw_move(i,xl,yl,spinsl):
    #counter = 0
    #while counter <1:
        #i = np.random.randint(0,len(x)-1)
        #j = np.random.choice(find_possible_points(x,y,spins))
    possible, positions = correct_x(x,y,i, spins)
    if possible:
        spinsl[positions[0]][positions[1]] = spinsl[xl[i]][yl[i]]
        spins[xl[i]][yl[i]] = 0
        xl[i] = positions[0]
        yl[i] = positions[1]
        #counter += 1
    plus_spin, min_spin = plus_min(spinsl)
    #min_spin
    #plus_spin_x, plus_spin_y = zip(*plus_spin)
    #min_spin_x, min_spin_y = zip(*min_spin)
    return spins,x,y

def metropolis_combo(spinsl,sweeps,T,xl,yl):
    # random choice of either a spin or saw move
    magnetization = np.zeros(sweeps)
    energy_saw = np.zeros(sweeps)
    length_x = np.zeros(sweeps)
    length_y = np.zeros(sweeps)
    activity = []
    possp = []
    # the loop is equiped with the code to 'reset' one of the subsystems, i.e. the spin subsystem.
    for sweep in range(sweeps):
        if np.random.randint(0,2) == 0: #spin move
        #    if sweep%(5*10**2)==0:
        #        config = np.ones((N,N))
        #        spins = spin_on_saw(config,x,y)
            temp_var = metropol_spin(spinsl, 1, T,xl,yl)
            #spins_copy = temp_var[0]
            magnetization[sweep] = temp_var[1]
            energy_saw[sweep] = temp_var[2]
            length_x[sweep] = max(x)-min(x)
            length_y[sweep] = max(y)-min(y)
            spinsl,xl,yl = temp_var[0], temp_var[3], temp_var[4]
            # 1D energy to be used as comparison to the actual energy
            #energy_1d[sweep] = energy_ising_1d(spins_1dising(spins,x,y))
        else: # path change
        #    if sweep%(5*10**2)==0:
        #        config = np.ones((N,N))
        #        spins = spin_on_saw(config,x,y)
            temp_var = metropol_saw(spins, 1, T,x,y)
            #spins_copy = temp_var[0]
            magnetization[sweep] = temp_var[1]
            energy_saw[sweep] = temp_var[2]
            length_x[sweep] = max(x)-min(x)
            length_y[sweep] = max(y)-min(y)
            spinsl,xl,yl = temp_var[0], temp_var[3], temp_var[4]
            activity.append(temp_var[5])
            possp = list(itertools.chain(possp,temp_var[6]))
            #print("combo")
            #print(activity)
            #print(possp)
    return magnetization, energy_saw, length_x, length_y, spinsl,xl,yl, activity,possp#, energy_1d

def create_adjacency(spin_array, N, x, y):
    adjacency = []
    for i in range(len(x)):
        temp_row = []
        for j in range(len(y)):
            if (((y[j] == y[i] - 1 or y[j] == y[i] + 1) and x[j] == x[i]) or (
                    (x[j] == x[i] - 1 or x[j] == x[i] + 1) and y[j] == y[i])) and spin_array[x[j]][y[j]] != 0:
                temp_row.append(1)
            else:
                temp_row.append(0)
        adjacency.append(temp_row)
    # print(len(adjacency))
    # print(len(adjacency[0]))
    return adjacency

def print_useful(adjacency):
    temp = '{'
    # print('{', end= '')
    for i in range(len(adjacency)):
        temp += '{'
        # print('{',end='')
        for j in range(len(adjacency[i]) - 1):
            temp += str(adjacency[i][j])
            temp += ","
            # print(adjacency[i][j],",", end = '')
        temp += str(adjacency[i][len(adjacency[i]) - 1])
        if i == len(adjacency) - 1:
            temp += '}'
        else:
            temp += '},'
        # print(adjacency[i][len(adjacency[i])-1],'},',end='')
    temp += '}\n'
    # print('}')
    return temp


# Initialize the SAW
n=50
'''GENERAL PATH'''
#x, y= saw(n)
#x= [10, 10, 9,8, 7, 7, 8, 9, 9, 10,11,11,11,11,11,10,9,9,10]
#y= [10, 9, 9, 9, 9, 10, 10, 10, 11, 11,11,10,9,8,7,7,7,8,8]
'''STRAIGHT PATH'''
x = [i for i in range(n,n+n)]
y = [n for _ in range(n,n+n)]
'''CURVED PATH'''
#x = [64,63,62,61,60,59,58,57,57,58,59,60,61,62,63,64,64,63,62,61,60,59,58,57,57,58,59,60,61,62,63,64,64,63,62,61,60,59,58,57,57,58,59,60,61,62,63,64,64,63,62,61,60,59,58,57,57,58,59,60,61,62,63,64]
#y = [64,64,64,64,64,64,64,64,63,63,63,63,63,63,63,63,62,62,62,62,62,62,62,62,61,61,61,61,61,61,61,61,60,60,60,60,60,60,60,60,59,59,59,59,59,59,59,59,58,58,58,58,58,58,58,58,57,57,57,57,57,57,57,57]
#x= [25,24,23,22, 21, 20, 20, 21, 22, 23,24,25,25,24,23,22,21,20,20, 21, 22, 23,24,25,25,24,23,22,21,20]
#y= [25,25,25,25, 25, 25, 24, 24, 24, 24,24,24,23,23,23,23,23,23,22, 22, 22, 22,22,22,21,21,21,21,21,21]
#x = [36,35,34,33,32,31,31,32,33,34,35,36,36,35,34,33,32,31,31,32,33,34,35,36,36,35,34,33,32,31,31,32,33,34,35,36]
#y = [36,36,36,36,36,36,35,35,35,35,35,35,34,34,34,34,34,34,33,33,33,33,33,33,32,32,32,32,32,32,31,31,31,31,31,31]

x_original = x.copy()
y_original = y.copy()
path = merge_lists(x,y)
N=5*n
'''SPIN CONFIGURATIONS (RANDOM OR ALL EQUAL)'''
config_initial = np.ones((N,N))#create_initial(N,N) #np.ones((N,N))
#dummy = config_initial.copy()
#config_initial[x[0]][y[0]] = 1
#config_initial[x[-1]][y[-1]] = 1
spins = spin_on_saw(config_initial,x,y)
spins_original = spins.copy()
spins_1d = spins_1dising(spins,x,y)
#spins = create_high_initial(spin_array,x,y)
#print(spins)
#print(config_initial)
#print(dummy)
'''POSSIBILITY FOR THE SPIN CONFIGURATION TO BECOME ALTERNATING'''
for i in range(len(x)):
    spins[x[i]][y[i]] = (-1)**i
spins_1d = spins_1dising(spins,x,y)

#config_initial[x[0]][y[0]] = 1
#config_initial[x[-1]][y[-1]] = 1
#spins_1d = spins_1dising(spins,x,y)
x_original = x.copy()
y_original = y.copy()
spins_original = spins.copy()

plus_spin, min_spin = plus_min(spins)
#min_spin
#plus_spin_x, plus_spin_y = zip(*plus_spin)
#min_spin_x, min_spin_y = zip(*min_spin)
'''ACTUAL SIMULATION (POSSIBLE TO USE TIME INTERVALS)'''
for j in range(21):
    print(j)
    spins = spins_original.copy()
    x = x_original.copy()
    y = y_original.copy()
    string_file_1 = "adjacency_flatnobc_{}.txt"
    formatted_string_file_1 = string_file_1.format(j)
    T=2
    steps = 10**3
    file1 = open(formatted_string_file_1, 'w')
    mag_after, energy_after, length_x, length_y, number_regions, energy_1d, activity= [] ,[],[],[], [], [], []
    poss = []
    file1.write(str(print_useful(create_adjacency(spins,N,x,y))))
    file1.write("\n")
    for i in range(1,101):
        t_b = time.time()
        number_regions.append(region_counter(spins,x,y))
        energy_1d.append(energy_ising_1d(spins_1dising(spins,x,y)))
        temp_var = metropolis_combo(spins,steps,T,x,y)
        mag_after = list(itertools.chain(mag_after,temp_var[0]))
        energy_after = list(itertools.chain(energy_after,temp_var[1]))
        length_x = list(itertools.chain(length_x,temp_var[2]))
        length_y = list(itertools.chain(length_y,temp_var[3]))
        activity = list(itertools.chain(activity, temp_var[7]))
        poss = list(itertools.chain(poss,temp_var[8]))
        spins = temp_var[4]
        x = temp_var[5]
        y = temp_var[6]
        t_e = time.time()
        Deltat = t_e-t_b
        #print(i, ":")
        #print(t_e-t_b, "s")
        print(i, ":",Deltat)
        #figure(i)
        temp_var = print_useful(create_adjacency(spins,N,x,y))
        #print(temp_var)
        # write the adjacency matrix to a file, this way it can be used to calculate the OR-curvature of the SAW at this moment in time.
        file1.write(str(temp_var))
        file1.write("\n")
    # write all variables to files 
    number_regions.append(region_counter(spins,x,y))
    energy_1d.append(energy_ising_1d(spins_1dising(spins,x,y)))
    file1.close()
    string_file_2 = "variables_flatnobc_{}.txt"
    formatted_string_file_2 = string_file_2.format(j)
    file_variables = open(formatted_string_file_2, "w")
    file_variables.write("magnetization:")
    file_variables.write("\n")
    file_variables.write(str(mag_after))
    file_variables.write("\n")
    file_variables.write("energy")
    file_variables.write("\n")
    file_variables.write(str(energy_after))
    file_variables.write("\n")
    file_variables.write("energy of the 1D case:")
    file_variables.write("\n")
    file_variables.write(str(energy_1d))
    file_variables.write("\n")
    file_variables.write("length both directions")
    file_variables.write("\n")
    file_variables.write(str(length_x))
    file_variables.write("\n")
    file_variables.write(str(length_y))
    file_variables.write("\n")
    file_variables.write("activity")
    file_variables.write("\n")
    file_variables.write(str(activity))
    file_variables.write("\n")
    file_variables.write("Number of regions")
    file_variables.write("\n")
    file_variables.write(str(number_regions))
    file_variables.write("\n")
    file_variables.write("All possible moves:")
    file_variables.write("\n")
    file_variables.write(str(poss))
    file_variables.close()

    print(np.mean(mag_after))
    print(np.mean(mag_after[-1000:]))
    print(np.mean(energy_after))
    print(np.mean(energy_after[-1000:]))
