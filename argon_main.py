#!/usr/bin/env python
# coding: utf-8
'''Autors: Annet Konings, Lorenzo Filipello '''


import argon_functions as fn

initialisation = True

while initialisation == True:
    print('This is a simulation of Argon particles, please choose one of the options below by pressing Y or N on your keyboard\n')
    print('Do you want to include an external constant electric field? Press Y for yes and N for no')
    get_library = input()
    if get_library == 'Y':
        import argon_functions_electric as fn
        print('What do you want to measure? Please digit the corresponding number')
        print('1. Plot the trajectories of the particles')
        print('2. Plot the energy of the particles')
        print('3. Plot the correlation function of the particles')
        get_input = input()
        
        if get_input == '1':
            print("Input the temperature (float):\n")
            get_temperature=float(input())
            print("Input the density (float):\n")
            get_density=float(input())
            print('Insert the number of timesteps (integer):\n')
            get_timesteps = int(input())
            print('Insert the values of the Electric field (float)')
            get_el_field = float(input())
            print('The code is running, please wait')
            fn.plot_trajectories(get_timesteps, get_temperature, get_density, get_el_field)
            break
            
        if get_input == '2':
            print("Input the temperature (float):\n")
            get_temperature=float(input())
            print("Input the density (float):\n")
            get_density=float(input())
            print('Insert the number of timesteps (integer):\n')
            get_timesteps = int(input())
            print('Insert the values of the Electric field (float)')
            get_el_field = float(input())
            print('The code is running, please wait')
            fn.plot_energy(get_timesteps, get_temperature, get_density, get_el_field)
            break
        
        if get_input == '3':
            print("Input the temperature (float):\n")
            get_temperature=float(input())
            print("Input the density (float):\n")
            get_density=float(input())
            print('Insert the values of the Electric field (float)')
            get_el_field = float(input())
            print('The code is running, please wait')
            fn.correlation_function(get_temperature, get_density, get_el_field)
            break
     
    if get_library == 'N':
        import argon_functions as fn
        print('What do you want to measure? Please digit the corresponding number')
        print('1. Plot the trajectories of the particles')
        print('2. Plot the energy of the particles')
        print('3. Measure the pressure of the particles')
        print('4. Plot the correlation function of the particles')
        get_input = input()
    
        if get_input == '1':
            print("Input the temperature (float):\n")
            get_temperature=float(input())
            print("Input the density (float):\n")
            get_density=float(input())
            print('Insert the number of timesteps (integer):\n')
            get_timesteps = int(input())
            print('The code is running, please wait')
            fn.plot_trajectories(get_timesteps, get_temperature, get_density)
            break
        if get_input == '2':
            print("Input the temperature (float):\n")
            get_temperature=float(input())
            print("Input the density (float):\n")
            get_density=float(input())
            print('Insert the number of timesteps (integer):\n')
            get_timesteps = int(input())
            print('The code is running, please wait')
            fn.plot_energy(get_timesteps, get_temperature, get_density)
            break
        if get_input == '3':
            print("Input the temperature (float):\n")
            get_temperature=float(input())
            print("Input the density (float):\n")
            get_density=float(input())
            print('The code is running, please wait')
            fn.get_pressure(get_temperature, get_density)
            break
        if get_input == '4':
            print("Input the temperature (float):\n")
            get_temperature=float(input())
            print("Input the density (float):\n")
            get_density=float(input())
            print('The code is running, please wait')
            fn.correlation_function(get_temperature, get_density)
            break
    

