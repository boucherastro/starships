import numpy as np

def madhu_seager(P, a1, a2, log_P1, log_P2, log_P3, T_set, P_set):
    '''
    Implementation from POSEIDON with very slight modification to make it 1D
    https://github.com/MartianColonist/POSEIDON/blob/main/POSEIDON/atmosphere.py
    
    Computes the temperature profile for an atmosphere using a re-arranged
    form of the P-T profile parametrisation in Madhusudhan & Seager (2009).

    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        a1 (float):
            Alpha_1 parameter (encodes slope in layer 1).
        a2 (float):
            Alpha_2 parameter encodes slope in layer 2).
        log_P1 (float):
            Pressure of layer 1-2 boundary.
        log_P2 (float):
            Pressure of inversion.
        log_P3 (float):
            Pressure of layer 2-3 boundary.
        T_set (float):
            Atmosphere temperature reference value at P = P_set (K).
        P_set (float):
            Pressure whether the temperature parameter T_set is defined (bar).
    
    Returns:
        T (np.array of float):
            Temperature of each layer as a function of pressure (K).
    
    '''

    # Store number of layers for convenience
    N_layers = len(P)
    
    # Initialise temperature array
    T = np.zeros(shape=(N_layers)) # 1D profile => N_sectors = N_zones = 1
    
    # Find index of pressure closest to the set pressure
    i_set = np.argmin(np.abs(P - P_set))
    P_set_i = P[i_set]
    
    # Store logarithm of various pressure quantities
    log_P = np.log10(P)
    log_P_min = np.log10(np.min(P))
    log_P_set_i = np.log10(P_set_i)
#     natural log/ln?
    # log_P = np.log(P)
    # log_P_min = np.log(np.min(P))
    # log_P_set_i = np.log(P_set_i)

    # By default (P_set = 10 bar), so T(P_set) should be in layer 3
    if (log_P_set_i >= log_P3):
        
        T3 = T_set  # T_deep is the isothermal deep temperature T3 here
        
        # Use the temperature parameter to compute boundary temperatures
        T2 = T3 - ((1.0/a2)*(log_P3 - log_P2))**2    
        T1 = T2 + ((1.0/a2)*(log_P1 - log_P2))**2    
        T0 = T1 - ((1.0/a1)*(log_P1 - log_P_min))**2   
        
    # If a different P_deep has been chosen, solve equations for layer 2...
    elif (log_P_set_i >= log_P1):   # Temperature parameter in layer 2
        
        # Use the temperature parameter to compute the boundary temperatures
        T2 = T_set - ((1.0/a2)*(log_P_set_i - log_P2))**2
        T1 = T2 + ((1.0/a2)*(log_P1 - log_P2))**2
        T3 = T2 + ((1.0/a2)*(log_P3 - log_P2))**2
        T0 = T1 - ((1.0/a1)*(log_P1 - log_P_min))**2   
        
    # ...or for layer 1
    elif (log_P_set_i < log_P1):  # Temperature parameter in layer 1
    
        # Use the temperature parameter to compute the boundary temperatures
        T0 = T_set - ((1.0/a1)*(log_P_set_i - log_P_min))**2
        T1 = T0 + ((1.0/a1)*(log_P1 - log_P_min))**2   
        T2 = T1 - ((1.0/a2)*(log_P1 - log_P2))**2  
        T3 = T2 + ((1.0/a2)*(log_P3 - log_P2))**2
        
    # Compute temperatures within each layer
    for i in range(N_layers):
        
        if (log_P[i] >= log_P3):
            T[i] = T3
        elif ((log_P[i] < log_P3) and (log_P[i] > log_P1)):
            T[i] = T2 + np.power(((1.0/a2)*(log_P[i] - log_P2)), 2.0)
        elif (log_P[i] <= log_P1):
            T[i] = T0 + np.power(((1.0/a1)*(log_P[i] - log_P_min)), 2.0)

    return T