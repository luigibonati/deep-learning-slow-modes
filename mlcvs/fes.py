import numpy as np

# FES
def compute_fes_1d(cv_x,logweights,kbt,sigma,extent,grid_bin=100,compute_derivatives=False,periodic=False,compute_deltaF=False,deltaF_max=0.):
    #define grid
    gx_min,gx_max = extent
    grid_x=np.linspace(gx_min,gx_max,grid_bin)
    
    #compute probability
    max_prob=0
    prob=np.zeros(grid_bin)
    deriv=[]
    if compute_derivatives:
        deriv=np.zeros(grid_bin)
    basinA,basinB=0,0
    period=0
    if periodic:
        period=gx_max-gx_min
    if periodic and compute_derivatives:
        print('derivatives not implemented for periodic variables')
        return None
        
    for i in range(grid_bin):
        dx,arg2=0,0
        if periodic:
            dx=np.absolute(grid_x[i]-cv_x)
            arg2=(np.minimum(dx,period-dx)/sigma)**2
        else:
            dx=grid_x[i]-cv_x
            arg2=(dx/sigma)**2
        prob[i]=np.sum(np.exp(logweights)*np.exp(-0.5*arg2))
        if compute_derivatives:
            deriv[i]=np.sum(-dx/(sigma**2)*np.exp(logweights)*np.exp(-0.5*arg2))
        if prob[i]>max_prob:
            max_prob=prob[i]
        if compute_deltaF:
            if grid_x[i]<deltaF_max:
                basinA+=prob[i]
            else:
                basinB+=prob[i]

    #convert to fes
    fes=-kbt*np.log(prob/max_prob)
    
    if compute_derivatives and compute_deltaF:
        deriv=-kbt*deriv/prob
        return grid_x,fes,deriv,deltaF
    elif compute_deltaF:
        deltaF=kbt*np.log(basinA/basinB)
        return grid_x,fes,deltaF
    else:
        return grid_x,fes

def compute_fes_2d(cv_x,cv_y,logweights,kbt,sigma,extent,grid_bin=100):
    #define grid
    gx_min,gx_max,gy_min,gy_max = extent
    xx=np.linspace(gx_min,gx_max,grid_bin)
    yy=np.linspace(gy_min,gy_max,grid_bin)
    grid_x,grid_y=np.meshgrid(xx,yy)

    #retrieve_sigma
    if isinstance(sigma, list):
        sigma_x,sigma_y=sigma
    else:
        sigma_x=sigma
        sigma_y=sigma
        
    #compute probability
    max_prob=0
    prob=np.zeros((grid_bin,grid_bin))
    for i in range(grid_bin):
        for j in range(grid_bin):
            dx=np.absolute(grid_x[i,j]-cv_x)
            dy=np.absolute(grid_y[i,j]-cv_y)
            arg2=(dx/sigma_x)**2+(dy/sigma_y)**2
            prob[i,j]=np.sum(np.exp(logweights)*np.exp(-0.5*arg2)) 
            if prob[i,j]>max_prob:
                max_prob=prob[i,j]

    #convert to fes
    fes=-kbt*np.log(prob/max_prob)
    
    return fes

def compute_deltaF_fes(fes,grid_x,transition_x=0,kbt=1):
    return kbt*np.log((np.exp(-fes[grid_x<transition_x]/kbt)).sum()/(np.exp(-fes[grid_x>transition_x]/kbt)).sum())
