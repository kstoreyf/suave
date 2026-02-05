import numpy as np 
import time

import Corrfunc
from Corrfunc.io import read_catalog
from Corrfunc.theory.DDsmu import DDsmu


print(Corrfunc.__file__)
print(Corrfunc.__version__)

def main():

    savetag = ''
    #proj_type = 'tophat'
    proj_type = 'piecewise'
    #ncomponents = 20
    ncomponents = 3
    r_edges = np.linspace(10., 100., ncomponents+1)
    #rmin = 0.1
    #rmax = 84.0
    #r_edges = np.logspace(np.log10(rmin),
    #                   np.log10(rmax),
    #                   ncomponents)
    proj_dict = {'tophat': {'ncomponents': ncomponents,
                            'proj_func': tophat_orig,
                            'proj_fn': None,
                            'args':[r_edges],
                            'kwargs':{}
                            },
                 'piecewise': {'ncomponents': ncomponents,
                            'proj_func': None,
                            'proj_fn': None,
                            'args':[r_edges],
                            'kwargs':{}
                            } 
                 }

    proj = proj_dict[proj_type]
    frac = 0.0001
    seed = 42
    allx, ally, allz = read_catalog()
    N = int(frac * len(allx))
    print("N:", N)
    np.random.seed(seed)
    x = np.random.choice(allx, N, replace=False)
    y = np.random.choice(ally, N, replace=False)
    z = np.random.choice(allz, N, replace=False)
    data = np.array([x,y,z]).T
    fmt='%10.10f'

    ### Brute force test
    # s = time.time()
    # print('brute force')
    # v_dd_correct, T_dd_correct = dd_bruteforce(data, proj['proj_func'], proj['ncomponents'], *proj['args'], **proj['kwargs'])
    # e = time.time()
    # print(v_dd_correct)
    # print(T_dd_correct)
    # print("brute force time:", e-s, 's')

    # Brute force w numpy trick
    # s = time.time()
    # print('numpy trick')
    # v_dd_correct, T_dd_correct = dd_bruteforce_numpy(data, proj['proj_func'], proj['ncomponents'], *proj['args'], **proj['kwargs'])
    # e = time.time()
    # print(v_dd_correct)
    # print(T_dd_correct)
    # print("numpy trick brute force time:", e-s, 's')

    # np.save(f'../output/correct_full_{proj_type}.npy', [v_dd_correct, T_dd_correct, proj_type, proj])
    # np.savetxt(f'../output/correct_vdd_{proj_type}.npy', v_dd_correct, fmt=fmt)
    # np.savetxt(f'../output/correct_Tdd_{proj_type}.npy', T_dd_correct, fmt=fmt)
    # print(v_dd_correct)
    # print(T_dd_correct)

    ### Corrfunc/suave AVX test
    periodic = True
    
    print('suave AVX')
    nthreads = 1
    mumax = 1.0
    nmubins = 1
    s = time.time()
    _, v_dd, T_dd = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, 
                proj_type=proj_type, ncomponents=proj['ncomponents'], projfn=proj['proj_fn'], periodic=periodic,
                isa='avx')#, c_api_timer=True)
    T_dd = T_dd.reshape((ncomponents, ncomponents)) #make code output it like this?! or maybe i didn't because it makes it easier to pass directly to compute_amps, etc
    e = time.time()
    print(v_dd)
    print(T_dd)
    #print(a)
    print("suave AVX time time:", e-s, 's')

    ### Corrfunc/suave test
    
    print('suave regular')
    #nthreads = 1
    #mumax = 1.0
    #nmubins = 1
    s = time.time()
    _, v_dd, T_dd = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                proj_type=proj_type, ncomponents=proj['ncomponents'], projfn=proj['proj_fn'], periodic=periodic,
                isa='fallback')#, c_api_timer=True)
    T_dd = T_dd.reshape((ncomponents, ncomponents)) #make code output it like this?! or maybe i didn't because it makes it easier to pass directly to compute_amps, etc
    e = time.time()
    print(v_dd)
    print(T_dd)
    #print(a)
    print("suave regular time:", e-s, 's')

    np.save(f'../output/suave_full_{proj_type}.npy', [v_dd, T_dd, proj_type, proj])
    np.savetxt(f'../output/suave_vdd_{proj_type}.npy', v_dd, fmt=fmt)
    np.savetxt(f'../output/suave_Tdd_{proj_type}.npy', T_dd, fmt=fmt)


def dd_bruteforce(data, proj_func, ncomponents, *args, **kwargs):
    # data is shape (N, 3)
    N = data.shape[0]
    v_dd = np.zeros(ncomponents)
    T_dd = np.zeros((ncomponents, ncomponents))
    
    for i in range(N): 
        for j in range(N):
            if i!=j:
                # this is the euclidean distance (https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy)
                r = np.linalg.norm(data[i]-data[j])
                u = proj_func(r, *args, **kwargs)
                v_dd += u
                T_dd += np.outer(u, u)
    return v_dd, T_dd

def dd_bruteforce_numpy(data, proj_func, ncomponents, *args, **kwargs):
    N = data.shape[0]
    v_dd = np.zeros(ncomponents)
    T_dd = np.zeros((ncomponents, ncomponents))
    dists = np.sqrt( -2 * np.dot(data, data.T) + np.sum(data**2, axis=1) + np.sum(data**2, axis=1)[:, np.newaxis] ) 
    print(dists.shape)
    us = np.empty((N, N, ncomponents))
    #proj_func_vect = np.vectorize(tophat_orig,otypes=[np.float],cache=False)
#j  %timeit list(vectfunc(lst_x,lst_y))
    #us = proj_func_vect(dists, *args, **kwargs)
    #us = proj_func(dists, *args, **kwargs)
    for i in range(N):
        for j in range(N):
            #if i!=j:
            if True:
                # this is the euclidean distance (https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy)
                r = dists[i,j]
                #r = np.linalg.norm(data[i]-data[j])
                u = proj_func(r, *args, **kwargs)
                v_dd += u
                T_dd += np.outer(u, u)
                #print(u)
                #us[i,j] = u
    #print(us.shape)
    #print(us)
    #v_dd = np.sum(us, axis=(0,1))
    #print(v_dd.shape)
    #T_dd = np.sum(np.outer(us, us.T), axis=0)
    return v_dd, T_dd

def tophat_orig(r, r_edges):
    u = np.zeros(len(r_edges)-1)
    for bb in range(len(r_edges)-1):
        if (r_edges[bb] <= r < r_edges[bb+1]):
            u[bb] = 1.0
            break
    return u

def tophat(r, r_edges):
    us = np.zeros(len(r_edges)-1)
    for bb in range(len(r_edges)-1):
        rmin = r_edges[bb]
        rmax = r_edges[bb+1]
        ubb = tophat_single(r, rmin, rmax)
        #print(rmin, rmax, ubb)
        us[bb] = np.sum(ubb, axis=0)
    return us

def tophat_single(r, rmin, rmax):
    #if (rmin < r < rmax):
    #    return 1    
    #print(np.where((r>=rmin) & (r<rmax)))
    in_bin = np.where((r>=rmin) & (r<rmax))[0]
    count = len(in_bin)
    #print(in_bin)
    #print(count)
    if count>0:
        print(in_bin, count)
    return count

def tophat_numpy(r, r_edges):
    n_bins = len(r_edges)-1
    u = np.zeros(n_bins)
    indices = np.digitize(r, r_edges)
    #print(indices, r, r_edges)
    #print(len(indices))
    #print(max(indices))
    # remove indices outside of bins
    indices = np.array(indices).flatten()
    #print(np.where((indices==0) | (indices==n_bins+1)))
    indices = np.delete(indices, np.where((indices==0) | (indices==n_bins+1)))
    #print(indices)
    # 0 means below the lowest bin for digitize; need to move all down
    indices -= 1
    #print(indices)
    #print(len(indices))
    #print(max(indices))
    u[indices] += 1
    return u
        

def tophat_numpy_big(us, r, r_edges):
    n_bins = len(r_edges)-1
    u = np.zeros(n_bikns)
    indices = np.digitize(r, r_edges)
    #print(indices, r, r_edges)
    # remove indices outside of bins
    indices = np.array(indices).flatten()
    #print(np.where((indices==0) | (indices==n_bins+1)))
    indices = np.delete(indices, np.where((indices==0) | (indices==n_bins+1)))
    #print(indices)
    # 0 means below the lowest bin for digitize; need to move all down
    indices -= 1
    #print(indices)
    #print(len(indices))
    #print(max(indices))
    u[indices] += 1
    return u


if __name__=='__main__':
    main()
