"""
Functions to create synthetic grid and data
case: modify the IEEE standard case reported in PyPower
data: modify the raw load and PV data to be suit for a specific case
"""

from optparse import Values
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import copy
from pypower.idx_bus import PD, QD, GS, BS
from pypower.idx_gen import QMAX, PMAX, QMIN
from pypower.idx_brch import RATE_A, BR_B, TAP, SHIFT
from pypower.api import ext2int, bustypes, case14
from os.path import exists
import sys
from tqdm import tqdm


"""
Improve the resolution of load and pv data
"""
def improve_resolution(load_raw, pv_raw, res):
    """
    Improved the resolution on raw data
    resolution: e.g. '5min'
    data: raw load
    """
    
    # data['DateTime'] = pd.to_datetime(data['DateTime'])
    # pv['DateTime'] = pd.to_datetime(pv['DateTime'])
    
    # if resolution == '5min':
    #     n = 2  # The number of new rows under each existing row 
    
    # """
    # data
    # """
    # # Construct a new dataframe
    # new_index = pd.RangeIndex(len(data)*(n+1))
    # # Put the old dataframe into the new dataframe
    # ids = np.arange(len(data))*(n+1)
    # data_new = pd.DataFrame(np.nan, index=new_index, columns=data.columns)
    # data_new.loc[ids] = data.values
    
    # """
    # PV
    # """
    # pv_new = pd.DataFrame(np.nan, index=new_index, columns=pv.columns)
    # pv_new.loc[ids] = pv.values
        
    # # Replace with the new time stamp
    # datetime_new = pd.date_range(start = data['DateTime'].iloc[0],
    #                             end = data['DateTime'].iloc[-1] + pd.DateOffset(minutes = 10),
    #                             freq = resolution)

    # data_new['DateTime'] = datetime_new
    # pv_new['DateTime'] = datetime_new
    # # Linear interpolation
    # data_new.interpolate(method = 'linear', inplace=True)
    # pv_new.interpolate(method = 'linear', inplace=True)
    
    # drop the time stamp and interpolate
    # Load
    load_raw.index = pd.to_datetime(load_raw.DateTime)
    load_raw.drop("DateTime", axis=1, inplace=True)
    load_new = load_raw.resample(res).interpolate()
    
    # PV
    pv_raw.index = pd.to_datetime(pv_raw.DateTime)
    pv_raw.drop("DateTime", axis=1, inplace = True)
    pv_new = pv_raw.resample(res).interpolate()
    
    return load_new, pv_new

"""
Add sharp change on pv data to mimic the cloud 
"""

def add_cloud(pv_new, unchange_rate, max_reduce):
    """
    unchange_rate: the possibility that a PV is not changed.
    max_reduce: the maximum PV reduction at a instance
    """
    
    seed = 1*(np.random.rand(pv_new.shape[0],pv_new.shape[1]-1) >= unchange_rate)
    reduce_rate = max_reduce*np.random.rand(pv_new.shape[0],pv_new.shape[1]-1)
    reduce = seed * reduce_rate
    value = pv_new.values[:,1:]*(1-reduce)

    pv_new.iloc[:,1:] = value

    return pv_new

"""
Modify the IEEE case
"""

def gen_case(case_name):
    """
    The function that is used to generate the load data.
    You have to write in this section by yourself if the case is not listed.
    In general, define the active load wisely in the self.case['bus'][:,PD] so that the generated load is not raising non-convergency in OPF

    case: an initiated case to be modified by some previous defined settings.
    Return: a new modified case.
    """

    if case_name == 'case14':
        """
        case14:
            1. Add load on the zero load non ref bus;
            2. Increase the load level;
        """
        case = case14()
        # Determine the bus type
        case_int = ext2int(case)
        ref_index, pv_index, pq_index = bustypes(case_int['bus'], case_int['gen'])  # reference bus (slack bus), pv bus, and pq (load bus)

        # Modify the load
        for i in range(len(case['bus'])):
            if case['bus'][i,PD] == 0 and i != ref_index:
                # Add load on the zero load non ref bus
                case['bus'][i,PD] = 30
                case['bus'][i,QD] = 0     # An arbitrary value
            
            #if case['bus'][i,PD] <= 15:
            #    # Increase the load on small loaded bus
            #    case['bus'][i,PD] = case['bus'][i,PD] * 2.
        
        # Increase the load at each bus
        case['bus'][:,PD] = case['bus'][:,PD] * 2.5
        
        # Increase QMAX in generator
        case['gen'][:,QMIN] = -case['gen'][:,QMAX]
        
        # Add constraints on the branch power flow RATE_A
        case['branch'][:,RATE_A] = case['branch'][:,RATE_A]/50  # The default is 9900           
        # Change the cost of the generators: a linear cost is considered
        # case['gencost'][:,3] = 2   # Linear cost
        # case['gencost'][:,4] = case['gencost'][:,5]  # Linear cost
        # case['gencost'][:,4] = 20
        # case['gencost'][:,5] = case['gencost'][:,6]  # Constant cost
    
    return case

"""
Generate load and pv for the specific system
"""

def gen_load(case, load_raw):
    """
    case: is a modified case file from, e.g. gen_case
    """
    
    load_active = []
    load_reactive = []
    for i in range(len(case['bus'])):
        # Active power
        load_active_ = load_raw.iloc[:,i+1].values
        load_active_max_ = np.max(load_active_)
        load_active_max = case['bus'][i,PD]     # rated power in the case file
        # Rescale
        load_active_ = load_active_ * load_active_max/load_active_max_
        load_active.append(load_active_)
        # Reactive power
        pf = 0.97+0.02*np.random.rand(len(load_raw)) # PF: 0.97-0.99
        Q_P = np.tan(np.arccos(pf))                       # Q to P ratio
        load_reactive_= load_active_*Q_P
        
        load_reactive.append(load_reactive_)
    
    load_active = np.array(load_active).T
    load_reactive = np.array(load_reactive).T
    
    return load_active, load_reactive

def gen_pv(pv_bus, pv_raw, load_active, penetration_ratio):
    
    pv = pv_raw.iloc[:,1:len(pv_bus)+1].values
    pv_max = np.max(np.sum(pv,axis=1))               # The maximum pv per instance
    load_max = np.max(np.sum(load_active, axis=1))   # The maximum load per instance
    
    pv_active = pv * penetration_ratio / pv_max * load_max
    pv_reactive = -np.tan(np.arccos(0.95)) * pv_active
        
    return pv_active, pv_reactive

def gen_measure(case, load_active, load_reactive, pv_active, pv_reactive, pv_bus):
    """
    Generate measurement
    case: the case class from SE
    Feed in ZERO PV if does not need
    """
    
    # Pad the PV bus
    pv_active_ = np.zeros((load_active.shape[0], load_reactive.shape[1]))
    pv_reactive_ = np.zeros((load_reactive.shape[0], load_reactive.shape[1]))
    pv_active_[:,pv_bus] = pv_active
    pv_reactive_[:,pv_bus] = pv_reactive
    
    z_noise_summary = []
    v_est_summary = []
    success_summary = []
    
    #for i in tqdm(range(len(load_active)-12*24,len(load_active))):
    for i in tqdm(range(len(load_active))):
        
        result = case.run_opf(load_active = load_active[i] - pv_active_[i], load_reactive = load_reactive[i] - pv_reactive_[i])
        z, z_noise, vang_ref, vmag_ref = case.construct_mea(result)
        success_summary.append(result['success'])

        if result['success'] == False:
            # The opf does not converge: Use the previous measurement to replace
            z_noise_summary.append(z_noise_summary[-1])
            v_est_summary.append(v_est_summary[-1])
            
        else:
            # The opf converges
            z_noise_summary.append(z_noise.flatten())
            # Do state estimation
            v_est = case.ac_se_pypower(z_noise=z_noise, vang_ref=vang_ref, vmag_ref=vmag_ref)
            v_est_summary.append(v_est)
        
        success_rate = np.sum(success_summary)/len(success_summary)
        if success_rate <= 0.97:
            print(f'Too many OPF does not converge. Reduce the load level.')
            break
    
    return z_noise_summary, v_est_summary, success_summary


if __name__ == "__main__":
    import sys
    sys.path.append('./')
    from configs.config import sys_config
    from configs.config_mea_idx import define_mea_idx_noise
    from utils.class_se import SE
    
    print(f'Generating measurement.')
    
    # Loading path
    noise_sigma_dir = f'gen_data\{sys_config["case_name"]}\\noise_sigma.npy'
    load_active_dir = f'gen_data\{sys_config["case_name"]}\load_active.npy'
    load_reactive_dir = f'gen_data\{sys_config["case_name"]}\load_reactive.npy'
    pv_active_dir = f'gen_data\{sys_config["case_name"]}\pv_active.npy'
    pv_reactive_dir = f'gen_data\{sys_config["case_name"]}\pv_reactive.npy'
    
    # Saving path
    z_noise_summary_dir = f'gen_data\{sys_config["case_name"]}\z_noise_summary.npy'
    v_est_summary_dir = f'gen_data\{sys_config["case_name"]}\\v_est_summary.npy'
    success_summary_dir = f'gen_data\{sys_config["case_name"]}\\success_summary.npy'
    
    # Modify case
    case = gen_case(sys_config['case_name'])
    idx, no_mea, noise_sigma = define_mea_idx_noise(case, choice = sys_config['measure_type'])
    noise_sigma = np.load(noise_sigma_dir)
    case_class = SE(case, noise_sigma, idx, fpr = sys_config['fpr'])
    
    # Generate measurement
    load_active = np.load(load_active_dir)
    load_reactive = np.load(load_reactive_dir)
    pv_active = np.load(pv_active_dir)
    pv_reactive = np.load(pv_reactive_dir)
    z_noise_summary, v_est_summary, success_summary = gen_measure(case = case_class, 
                                                                load_active = load_active, 
                                                                load_reactive = load_reactive, 
                                                                pv_active = pv_active, 
                                                                pv_reactive=0, 
                                                                pv_bus=sys_config['pv_bus'])
    
    np.save(z_noise_summary_dir, z_noise_summary, allow_pickle=True)
    np.save(v_est_summary_dir, v_est_summary, allow_pickle=True)
    np.save(success_summary_dir, success_summary, allow_pickle=True)