#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d



def distance(positions, j,i,t): 
'''This function takes the position vectors of two particles at time t and returns their distance in scalar and vector form and the vector  according to minimal image convention'''
    direction = np.zeros(3)
    for l in range(3):
        direction[l] = (positions[l,t,j] - positions[l,t,i] + box_size/2) % box_size - box_size/2 #Minimal image convention
    radial_distance = np.sqrt(np.sum(direction[0]**2 + direction[1]**2 + direction[2]**2))

    return radial_distance, direction

def LJ_pot(r):
'''This function takes the scalar distance of two particles and returns their Lennard-Jones potential'''
    LJ_potential = 4*(r**(-12) - r**(-6))
    return LJ_potential

def grad_LJ_pot(r):
'''This function takes the scalar distance of two particles and returns the gradient of their Lennard-Jones potential'''
    grad_LJ_potential = 4*(6*r**(-7) - 12*r**(-13))
    return grad_LJ_potential

def interaction(positions,j,t): 
'''This function evaluates the resulting forces acting on particle j at time t'''
    Force_x,Force_y,Force_z = 0,0,0
    for i in range(108):
        if i!=j:
            rad_dist, direction = distance(positions,j,i,t)
            Force_x += -grad_LJ_pot(rad_dist) * direction[0]/rad_dist
            Force_y += -grad_LJ_pot(rad_dist) * direction[1]/rad_dist
            Force_z += -grad_LJ_pot(rad_dist) * direction[2]/rad_dist
            
    return Force_x, Force_y, Force_z


    


def initial_conditions(temperature, density):
    
    particles = np.zeros((108,6))   
    l=-1 # Setting up the FCC lattice
    for i in range(3):
        for j in range(3):
            for k in range(3):
                l+=1
                # particle nr. 1
                particles[l,0]=0+box_size/3*i 
                particles[l,1]=0+box_size/3*j 
                particles[l,2]=0+box_size/3*k 
                # particle nr. 2
                particles[l+27, 0]=box_size/6+box_size/3*i
                particles[l+27, 1]=box_size/6+box_size/3*j
                particles[l+27, 2]=0+box_size/3*k
                # particle nr. 3
                particles[l+54,0]=box_size/6+box_size/3*i 
                particles[l+54,1]=0+box_size/3*j
                particles[l+54,2]=box_size/6+box_size/3*k 
                # particle nr. 4
                particles[l+81,0]=0+box_size/3*i
                particles[l+81,1]=box_size/6+box_size/3*j
                particles[l+81,2]=box_size/6+box_size/3*k
    
    
    v_x = np.zeros((1,108))
    v_y = np.zeros((1,108))
    v_z = np.zeros((1,108))
    v_x[0] = np.random.normal(0,np.sqrt(temperature),108)
    v_y[0] = np.random.normal(0,np.sqrt(temperature),108)
    v_z[0] = np.random.normal(0,np.sqrt(temperature),108)
        
       
    particles[:,3:] = np.array([v_x,v_y,v_z]).reshape(108,3)
    
    return particles

def verlet_alg(timesteps,initial_particles):
   
    x = np.zeros((timesteps,108)) 
    y = np.zeros((timesteps,108)) 
    z = np.zeros((timesteps,108)) 
    
    v_x = np.zeros((timesteps,108)) 
    v_y = np.zeros((timesteps,108)) 
    v_z = np.zeros((timesteps,108)) 
    
    F_x=np.zeros((timesteps,108))
    F_y=np.zeros((timesteps,108))
    F_z=np.zeros((timesteps,108))
    
    positions= np.zeros((3,timesteps,108))
    
    x[0,:]=initial_particles[:,0]
    y[0,:]=initial_particles[:,1]
    z[0,:]=initial_particles[:,2]
    v_x[0,:]=initial_particles[:,3]
    v_y[0,:]=initial_particles[:,4]
    v_z[0,:]=initial_particles[:,5]
    
    positions[0,:,:] = x
    positions[1,:,:] = y
    positions[2,:,:] = z
    
    h = 0.001

    for i in range(108):
        F_x[0,i],F_y[0,i],F_z[0,i] = interaction(positions,i,0)
    
    for t in range(0,timesteps-1): 
        for i in range(108):
            x[t+1,i] = (x[t,i] + v_x[t,i]*h + F_x[t,i]/2*h**2) % box_size
            positions[0,t+1,i] = x[t+1,i]
            
            y[t+1,i] = (y[t,i] + v_y[t,i]*h + F_y[t,i]/2*h**2) % box_size
            positions[1,t+1,i] = y[t+1,i]
            
            z[t+1,i] = (z[t,i] + v_z[t,i]*h + F_z[t,i]/2*h**2) % box_size
            positions[2,t+1,i] = z[t+1,i]
                      
        for i in range(108):
            F_x[t+1,i],F_y[t+1,i],F_z[t+1,i] = interaction(positions,i,t+1)
        
        for i in range(108):
            v_x[t+1,i] = (F_x[t,i]+F_x[t+1,i])*h/2 + v_x[t,i]
            v_y[t+1,i] = (F_y[t,i]+F_y[t+1,i])*h/2 + v_y[t,i]
            v_z[t+1,i] = (F_z[t,i]+F_z[t+1,i])*h/2 + v_z[t,i]

    velocities = np.array([v_x, v_y, v_z])

    return positions, velocities

def kin_energy(v_x,v_y,v_z,t):

    kin_total=0
    
    for i in range(108):
        En_kin = 0.5 * (v_x[t-1,i]**2+v_y[t-1,i]**2+v_z[t-1,i]**2)
        kin_total += En_kin
        
    return kin_total
    

def relaxation(temperature, density):
    
    timesteps=10
    iterations = 10
    
    initial_particles = initial_conditions(temperature, density)
    init_pos = initial_particles[:,:3]
    init_vel = initial_particles[:,3:]
    
   
    for i in range(iterations):
        positions , velocities = verlet_alg(timesteps,initial_particles)
        E_kin=kin_energy(velocities[0],velocities[1],velocities[2],timesteps)                     
        
        lmbd = np.sqrt((107*3*temperature)/(2*E_kin))
        
        for i in range(108):
            for l in range(3):
                initial_particles[i,l] = positions[l,timesteps-1,i]
                initial_particles[i,l+3] = lmbd*velocities[l,timesteps-1,i]

    return initial_particles



def argon_simulation(timesteps, temperature, density):
    global box_size
    box_size = (108/density)**(1/3) # Box size
    
    rescaled_particles = relaxation(temperature, density)  
    positions, velocities = verlet_alg(timesteps,rescaled_particles)
    
    return positions, velocities

def energy(positions,velocities,t):
    
    kinetic_en = 0
    potential_en = 0
    total_en = 0
    for i in range(108):
        potential=0
        kinetic = 0.5 * (velocities[0,t,i]**2+velocities[1,t,i]**2+velocities[2,t,i]**2)
        for j in range(108):
            if j!=i:
                dist,direction = distance(positions,j,i,t)
                potential += LJ_pot(dist)/2
                
        total = kinetic + potential
        total_en += total
        kinetic_en += kinetic
        potential_en += potential
    
    return kinetic_en, potential_en, total_en

def energy_arrays(positions,velocities, timesteps):

    en_tot = np.zeros(timesteps)
    en_kin = np.zeros(timesteps)
    en_pot = np.zeros(timesteps)
    
    for i in range(timesteps):
        en_kin[i], en_pot[i], en_tot[i] = energy(positions,velocities,i)
    
    return en_kin, en_pot, en_tot,


def pair_corelation(positions,t):
 
    pair_distances=np.empty((0,0))
    
    for i in range(108):
        for j in range(108):
            dist, direction = distance(positions,i,j,t)
            if dist<box_size/2 and i<j:
                pair_distances = np.append(pair_distances,dist)
            
    return pair_distances

def average_correlation(temperature,density):

    timesteps = 100
    iterations = 5
    
    avg_correlation = np.zeros(50)
    
    
    for i in range(iterations):
        positions, velocities = argon_simulation(timesteps, temperature, density)
        bin_number = np.arange(0,box_size/2,box_size/102)
        for t in range(timesteps-10,timesteps-1,1):
            pair_dist_list = pair_corelation(positions,t)
            histogram,bins = np.histogram(pair_dist_list, bins=bin_number)
            avg_correlation += histogram
    average_corr_normalized = avg_correlation/(iterations*len(range(timesteps-10,timesteps-1,1)))

    
    f = plt.figure(1)
    plt.bar(np.arange(0,box_size/2,box_size/100), average_corr_normalized) 
    plt.xlabel('Radial distance')
    plt.ylabel('Number of particles')
    plt.title(r'Average correlation for $\rho$ = {}'.format(density) + r' and $T$ = {}'.format(temperature))
    plt.savefig('avg_n.png')
    f.show()

    
    return average_corr_normalized
  
def correlation_function(temperature,density):
   

    average_pair_corr = average_correlation(temperature,density)
    
    corr_function = []
    
    start_int = box_size/100
    end_int = box_size/2 - start_int

    r=np.linspace(start_int, end_int, 49)

    for i in range(1,len(average_pair_corr)):
        g = (2*(box_size)**3)/(108*107*4*np.pi*r[i-1]**2*(box_size/100)) * average_pair_corr[i]
        corr_function.append(g)

    r_new=np.linspace(box_size/100,box_size/2,500)
    g_new=interp1d(r,corr_function, kind = 'quadratic', fill_value="extrapolate")(r_new)

    g = plt.figure(2)
    plt.scatter(r,corr_function)
    plt.plot(r_new,g_new)
    plt.ylabel('g(r)') 
    plt.xlabel('r')
    plt.title(r'Correlation function for $\rho$ = {}'.format(density) + r' and $T$ = {}'.format(temperature))
    plt.savefig('corr_func.png')
    g.show()
    
    return corr_function, r


def virial_theorem(temperature,density):
   

    timesteps = 100
    positions, velocities = argon_simulation(timesteps, temperature, density)
    r_list=[]
    virial=0
    for t in range(timesteps-10, timesteps-1, 1):
        for i in range(108):
            for j in range(108):
                if j>i:
                    dist, direction = distance(positions,i,j,t)
                    r_list.append(dist)
                    virial+=0.5*dist*grad_LJ_pot(dist)
    time_aver_virial = virial/len(range(timesteps-10, timesteps-1, 1))
    return time_aver_virial

def get_pressure(temperature,density):
    
    iterations = 10
    
    pressure_list = [] 
    average_virial = 0
    for i in range(iterations):
        vir = virial_theorem(temperature,density)
        pressure_single_simulation = temperature*density-density*vir/(108*3)
        pressure_list.append(pressure_single_simulation)
        average_virial += vir/iterations
           
    pressure = temperature*density-density/(108*3) * average_virial
    pressure_std=np.std(pressure_list)
    
    pressure_to_dataframe = {'pressure':pressure_list, 'density':density, 'temperature':temperature}
    pressure_df = pd.DataFrame(data = pressure_to_dataframe)

    pressure_df.to_csv('pressure.zip')
    return pressure_list, pressure, pressure_std

def plot_energy(timesteps, temperature, density):
    
    positions, velocities = argon_simulation(timesteps, temperature, density)
    kin_en, pot_en, tot_en = energy_arrays(positions, velocities, timesteps)
    
    plt.figure()
    plt.plot(kin_en/108, label = 'Kin. En.')
    plt.plot(pot_en/108, label = 'Pot. En.')
    plt.plot(tot_en/108, label = 'Tot. En.')
    plt.xlabel('timesteps')
    plt.ylabel('energy')
    plt.legend()
    plt.title(r'Average energy per particle for $\rho$ = {}'.format(density) + r' and $T$ = {}'.format(temperature))
    plt.savefig('argon_energy.png')
    plt.show()
    return kin_en, pot_en, tot_en

def plot_trajectories(timesteps, temperature, density):
    
    positions, velocities = argon_simulation(timesteps, temperature, density)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(108):
        ax.scatter(positions[0,:,i], positions[1,:,i], positions[2,:,i], s= 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(r'Particles trajectory for $\rho$ = {}'.format(density) + r' and $T$ = {}'.format(temperature))
    plt.savefig('particles_path.png')
    plt.show()
    return positions, velocities

