import torch
import numpy as np

from . import utils
from . import nncv

'''###############################
   #       NN architecture       #
   ###############################'''

class DeepTICA(nncv.BaseNNCV):
    ''' Neural Network class for DeepTica '''

    def __init__(self, layers , activation='tanh', output='linear', beta_output=1.):
        '''
        Define a NN object given the list of layers

        Parameters:
            layers (list): number of neurons per layer

            activation (string): type of activation function
                - relu (default)
                - tanh
                - elu
                - linear

        TODO: add doc for softmax and sigmoid

        Returns:
            nn.Module
        '''
        super().__init__()

        #get activation function
        activ=None
        if   activation == 'relu':
            activ=torch.nn.ReLU(True)
        elif activation == 'elu':
            activ=torch.nn.ELU(True)
        elif activation == 'tanh':
            activ=torch.nn.Tanh()
        #elif activation == 'silu':
        #    activ=SiLU()
        elif activation == 'linear':
            print('WARNING: linear activation selected')
        else:
            raise ValueError("unknown activation. options: 'relu','elu','tanh','linear'. ")

        #Create architecture
        modules=[]
        for i in range( len(layers)-1 ):
            if( i<len(layers)-2 ):
                modules.append( torch.nn.Linear(layers[i], layers[i+1]) )
                if activ is not None:
                    modules.append( activ )
            else:
                modules.append( torch.nn.Linear(layers[i], layers[i+1]) )

        #save output activation
        self.output = output

        # nn
        self.nn = torch.nn.Sequential(*modules)
        # hidden layer average
        self.average = torch.nn.Parameter(torch.zeros(layers[-1]), requires_grad=False)
        # tica evals and evecs
        self.evals_ = torch.nn.Parameter(torch.zeros(layers[-1]), requires_grad=False)
        self.evecs_ = torch.nn.Parameter(torch.zeros(layers[-1],layers[-1]), requires_grad=False)

        # n_input 
        self.n_input = layers[0]
        
        # options
        self.normIn = False
        self.normOut = False
        self.output_hidden = False

        # set temperature parameter for output activation
        self.beta_output = beta_output
        
        #set type and device
        self.dtype_ = torch.float32
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device_)
        
        #regularization
        self.reg_cholesky = None
        
        #optimizer
        self.opt_ = None
        self.earlystopping_ = None
        self.lrscheduler_ = None

        #training logs
        self.epochs = 0
        self.loss_train = []
        self.loss_valid = []
        self.evals_train = []
        self.log_header = True

    def get_params(self):
        out = dict()
        out['device']=self.device_
        out['Early Stopping']=True if self.earlystopping_ is not None else False
        out['LR scheduler']=True if self.lrscheduler_ is not None else False
        out['Input standardization']=self.normIn
        out['Output standardization']=self.normOut
        out['# epochs']=self.epochs
        return out
    
    def get_hidden(self, x: torch.Tensor) -> (torch.Tensor):
        '''
        Forward pass, computes NN output.

        Parameters:
            x (torch.Tensor): input

        Returns:
            z (torch.Tensor): NN output
        '''
        if self.normIn:
            x = self._normalize(x,self.MeanIn,self.RangeIn)
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        z = self.nn(x)
        #nonlinear activation function in the output
        if self.output == 'softmax':
            z = self._apply_softmax(z).clone() # to avoid in-place operation
        elif self.output == 'sigmoid':
            z = self._apply_sigmoid(z).clone() # to avoid in-place operation

        return z

    def _apply_tica(self, x: torch.Tensor) -> (torch.Tensor):

        z = torch.matmul(x,self.evecs_)
        return z

    def _apply_softmax(self, x: torch.Tensor) -> (torch.Tensor):
            z = torch.nn.functional.softmax( torch.mul(x, self.beta_output ) , dim=1)
            return z

    def _apply_sigmoid(self, x: torch.Tensor) -> (torch.Tensor):
            z = torch.sigmoid( torch.mul(x, self.beta_output ) )
            return z

    def set_cholesky_regularization(self, x):
        '''
        TODO
        '''
        self.reg_cholesky = x
        
    def set_average(self, x: torch.Tensor):
        '''
        Set (weighted) average of the hidden components.

        Parameters:
            x (tensor): average
        '''
        self.average = torch.nn.Parameter(x, requires_grad=False)

    def get_average(self) -> (torch.Tensor):
        '''
        Get average of the hidden components.

        Returns:
            x (tensor): tica eigenvectors
        '''
        return self.average

    def _remove_average(self, x: torch.Tensor) -> (torch.Tensor):
        ''' Obtain mean-free hidden variables '''
        z = x.sub(self.average)
        return z

    def forward(self, x: torch.Tensor) -> (torch.Tensor):
        '''
        Compute DeepTica projections.
        The tica eigenvectors and the average must be set beforehand.

        Parameters:
            x (tensor): input

        Returns:
            x (tensor): deep-tica CV
        '''
        z = self.get_hidden(x)
        z = self._remove_average(z)
        z = self._apply_tica(z)
        if self.normOut:
            z = self._normalize(z,self.MeanOut,self.RangeOut)

        return z

    def compute_correlation_matrix(self,x,y,w):
        '''
        Compute the correlation matrix between x and y with weights w

        Parameters:
            x (tensor): first array
            y (tensor): second array
            w (float): weights

        Returns:
            corr (tensor): correlation matrix
        '''
        d = x.shape[1]
        #define arrays
        corr = torch.zeros((d,d))
        norm = torch.sum(w)
        #compute correlation matrix
        corr = torch.einsum('ij, ik, i -> jk', x, y, w )
        corr /= norm
        corr = 0.5*(corr + corr.T)

        return corr

    #tica eigenvalues and eigenvectors
    def solve_tica_eigenproblem(self,C_0,C_lag,n_eig=0,save_params=False):
        '''
        Compute generalized eigenvalue problem : C_lag * wi = lambda_i * C_0 * w_i

        Parameters:
            C_0 (tensor): correlation matrix at lag-time 0
            C_lag (tensor): correlation matrix at lag-time lag
            scipy (bool): wheter to use scipy.eigh to compute eigenvecs (debug only)

        Returns:
            eigvals (tensor): eigenvalues in ascending order
            #eigvecs (tensor): matrix whose column eigvecs[:,i] is eigenvector associated to eigvals[i]
        '''
        #cholesky decomposition
        if self.reg_cholesky is not None:
            L = torch.cholesky(C_0+self.reg_cholesky*torch.eye(C_0.shape[0]),upper=False)
        else:
            L = torch.cholesky(C_0,upper=False)
            
        L_t = torch.t(L)
        L_ti = torch.inverse(L_t)
        L_i = torch.inverse(L)
        C_new = torch.matmul(torch.matmul(L_i,C_lag),L_ti)

        #find eigenvalues and vectors of C_new
        eigvals, eigvecs = torch.symeig(C_new,eigenvectors=True)
        #sort
        eigvals, indices = torch.sort(eigvals, 0, descending=True)
        eigvecs = eigvecs[:,indices]
        
        #return to original eigenvectors
        eigvecs = torch.matmul(L_ti,eigvecs)

        #normalize them
        for i in range(eigvecs.shape[1]): # maybe change in sum along axis?
            norm=eigvecs[:,i].pow(2).sum().sqrt()
            eigvecs[:,i].div_(norm)
        #set the first component positive
        eigvecs.mul_( torch.sign(eigvecs[0,:]).unsqueeze(0).expand_as(eigvecs) )

        #keep only first n_eig eigvals and eigvecs
        if n_eig>0:
            eigvals = eigvals[:n_eig]
            eigvecs = eigvecs[:,:n_eig]
        
        if save_params:
            self.evals_ = torch.nn.Parameter(eigvals.detach(), requires_grad=False) 
            self.evecs_ = torch.nn.Parameter(eigvecs.detach(), requires_grad=False) 
    
        return eigvals

    '''###############################
       #   Aux. training functions   #
       ###############################'''

    def forward_and_tica(self, data, n_eig=0, return_corr=False, save_params=False):
        '''
        1) Calculate the NN output
        2) Remove average
        3) Compute TICA

        params:
            model: NN model
            data: batch of configurations (x_t, x_lag, w_t, w_lag)
            set_tica (bool): save the tica eigen and the average of h in the model
            n_eig (int): save in the model only the first n_eig eigenvectors

        return:
            eigvals,eigvecs
        '''
        # =================get data===================
        x_t, x_lag, w_t, w_lag = data   
        # =================forward====================
        f_t = self.get_hidden(x_t)
        f_lag = self.get_hidden(x_lag)
        # =============compute average================
        ave_f = torch.einsum('ij,i ->j',f_t,w_t)/torch.sum(w_t)
        # ==============remove average================
        f_t.sub_(ave_f)
        f_lag.sub_(ave_f)
        # ===================tica=====================
        C_0 = self.compute_correlation_matrix(f_t,f_t,w_t)
        C_lag = self.compute_correlation_matrix(f_t,f_lag,w_lag)
        evals = self.solve_tica_eigenproblem(C_0,C_lag,n_eig=n_eig,save_params=save_params) 
        # ================set values==================
        if save_params:
            self.set_average(ave_f)

        if return_corr:
            return evals,C_lag
        else:
            return evals

    #define loss function
    #TODO add number of eigenvalues / vectors for GAP
    def loss_function(self,evals,objective='sum2',n_eig=0):
        '''
        Calculate loss function given the eigenvalues.

        params:
            evals: eigenvalues of TICA

            objective: function to optimize
                - sum     : -sum_i (lambda_i)
                - sum2    : -sum_i (lambda_i)**2
                - gap     : -(lambda_1-lambda_2)
                - gapsum  : -sum_i (lambda_{i+1}-lambda_i)
                - time    : -sum_i (1/log(lambda_i))
                - single  : - (lambda_i)
                - single2 : - (lambda_i)**2

            n_eig: number of eigenvalues to include in the loss (default: 0 --> all). in case of single and single2 is used to specify which eigenvalue to use.


        returns:
            scalar loss
        '''
        #check if n_eig is given and
        if (n_eig>0) & (len(evals) < n_eig):
            raise ValueError("n_eig must be lower than the number of eigenvalues.")
        elif (n_eig==0):
            if ( (objective == 'single') | (objective == 'single2')):
                raise ValueError("n_eig must be specified when using single or single2.")
            else:
                n_eig = len(evals)
        elif (n_eig>0) & (objective == 'gapsum') :
            raise ValueError("n_eig parameter not valid for gapsum. only sum of all gaps is implemented.")

        loss = None
        
        if   objective == 'sum':
            loss = - torch.sum(evals[:n_eig])
        elif objective == 'sum2':
            g_lambda = - torch.pow(evals,2)
            loss = torch.sum(g_lambda[:n_eig])
        elif objective == 'gap':
            loss = - (evals[0] -evals[1])
        #elif objective == 'gapsum':
        #    loss = 0
        #    for i in range(evals.shape[0]-1):
        #        loss += - (evals[i+1] -evals[i])
        elif objective == 'time':
            g_lambda = 1 / torch.log(evals)
            loss = torch.sum(g_lambda[:n_eig])
        elif objective == 'single':
            loss = - evals[n_eig-1]
        elif objective == 'single2':
            loss = - torch.pow(evals[n_eig-1],2)
        else:
            raise ValueError("unknown objective. options: 'sum','sum2','gap','single','time'.")

        return loss

    def train_epoch(self, dataset, loss_type, n_eig):
        for batch in dataset: 
            # ===================tica=====================
            evals = self.forward_and_tica(batch,n_eig=n_eig,save_params=True) 
            # ===================loss===================== 
            loss = self.loss_function(evals,objective=loss_type,n_eig=n_eig)
            # =================backprop===================
            self.opt_.zero_grad()
            loss.backward()
            self.opt_.step()
        # ===================log======================
        self.epochs +=1 
    
    def evaluate_dataset(self, data, loss_type, n_eig, save_params=False):   
        with torch.no_grad():
            evals = self.forward_and_tica(data,n_eig=n_eig,save_params=save_params)
            loss = self.loss_function(evals,objective=loss_type,n_eig=n_eig)       
        return loss
    
    #TODO update if valid_data is not present
    
    def train(self, train_data, valid_data=None, standardize_inputs = True, standardize_outputs = True, loss_type='sum2', n_eig=0, nepochs=1000, log_every=1, info=False):
        
        #copy of all training set
        all_train = [torch.cat([batch[i] for batch in train_data]) for i in range(4)]
        
        #check optimizer
        if self.opt_ is None:
            self.default_optimizer()

        #standardize inputs
        if standardize_inputs: #average from all training configs
            self.standardize_inputs(all_train[0])
            
        #print info
        if info:
            self.print_info()

        # -- Training --
        for ep in range(nepochs):
            self.train_epoch(train_data,loss_type,n_eig)
            
            loss_train = self.evaluate_dataset(all_train, loss_type, n_eig)
            loss_valid = self.evaluate_dataset(valid_data, loss_type, n_eig)
            self.loss_train.append(loss_train)
            self.loss_valid.append(loss_valid)
            self.evals_train.append(torch.unsqueeze(self.evals_,0))
            
            #standardize output
            if standardize_outputs:
                self.standardize_outputs(all_train[0])
            
            #earlystopping and lrschedule
            if self.lrscheduler_ is not None:
                self.lrscheduler_(loss_valid)
            if self.earlystopping_ is not None:
                self.earlystopping_(loss_valid,model=self.parameters,epoch=self.epochs)
            
            #log
            if ((ep+1) % log_every == 0) or ( self.earlystopping_.early_stop ):
                self.print_log({'Epoch':self.epochs,'Train Loss':loss_train,'Valid Loss':loss_valid,'Eigenvalues':self.evals_},
                               spacing=[6,12,12,24],decimals=4)
            
            #check whether to stop
            if (self.earlystopping_ is not None) and (self.earlystopping_.early_stop):
                self.parameters = self.earlystopping_.best_model
                break
        
    def export(self,tr_folder): #model,opt,evals,evecs,colvar_train,ep,loss_tica_train,loss_tica_valid):

        #!mkdir -p "{tr_folder}"
        # == EXPORT CHECKPOINT ==
        torch.save({
                    'epoch': self.epochs,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.opt_.state_dict(),
                    }, tr_folder+"checkpoint")

        print("@@ checkpoint: ",tr_folder+"torch_checkpoint.pt")

        # == Export jit model ==
        fake_input = torch.zeros(self.n_input).reshape(1,self.n_input)
        mod = torch.jit.trace(self, fake_input)
        mod.save(tr_folder+"model_all.pt")
        print("@@ traced torchscript model (for C++) in: ", tr_folder+"model_deeptica.pt" )

        # == Export model with single CV ==
        #for i in range(0,len(self.evals_)):
        #    print('@@ Exporting no. '+str(i))
        #    model.set_tica(torch.unsqueeze(evecs[:,i],1))
        #    model.standardize_outputs(torch.tensor(colvar_train,dtype=dtype,device=device))
        #    mod = torch.jit.trace(model, fake_input)
        #    mod.save(tr_folder+"model_"+str(i)+".pt")
        #    print("["+str(i)+ "] traced torchscript model (for C++) in: ", tr_folder+"model_deeptica"+str(i)+".pt" )

        #restore situation with all cvs
        #model.set_tica(evecs[:,-n_eig_loss:])
        #model.standardize_outputs(torch.tensor(colvar_train,dtype=dtype,device=device))

        # TODO
        # == EXPORT SUMMARY == 
        #myfile = open(tr_folder+'training-setup.txt', 'w')
        #myfile.write()
        #myfile.close()

'''###############################
   #      Training dataset       #
   ###############################'''

from bisect import bisect_left

def closest_idx_torch(array, value):
        '''
        Find index of the element of 'array' which is closest to 'value'.
        The array is first converted to a np.array in case of a tensor.
        Note: it does always round to the lowest one.

        Parameters:
            array (tensor/np.array)
            value (float)

        Returns:
            pos (int): index of the closest value in array
        '''
        if type(array) is np.ndarray:
            pos = bisect_left(array, value)
        else:
            pos = bisect_left(array.numpy(), value)
        if pos == 0:
            return 0
        elif pos == len(array):
            return -1
        else:
            return pos-1

def look_for_configurations(x,t,lag):
    '''
    Searches for all the pairs which are distant 'lag' in time, and returns the weights associated to lag=lag as well as the weights for lag=0.

    Parameters:
        x (tensor): array whose columns are the descriptors and rows the time evolution
        time (tensor): array with the simulation time
        lag (float): lag-time

    Returns:
        x_t (tensor): array of descriptors at time t
        x_lag (tensor): array of descriptors at time t+lag
        w_t (tensor): weights at time t
        w_lag (tensor): weights at time t+lag
    '''
    #define lists
    x_t = []
    x_lag = []
    w_t = []
    w_lag = []
    #find maximum time idx
    idx_end = closest_idx_torch(t,t[-1]-lag)
    start_j = 0
    #loop over time array and find pairs which are far away by lag
    for i in range(idx_end):
        stop_condition = lag+t[i+1]
        n_j = 0
        for j in range(i,len(t)):
            if ( t[j] < stop_condition ) and (t[j+1]>t[i]+lag):
                x_t.append(x[i])
                x_lag.append(x[j])
                deltaTau=min(t[i+1]+lag,t[j+1]) - max(t[i]+lag,t[j])
                w_lag.append(deltaTau)
                if n_j == 0: #assign j as the starting point for the next loop
                    start_j = j
                n_j +=1
            elif t[j] > stop_condition:
                break
        for k in range(n_j):
            w_t.append((t[i+1]-t[i])/float(n_j))

    x_t = torch.Tensor(x_t)
    x_lag = torch.Tensor(x_lag)
    w_t = torch.Tensor(w_t)
    w_lag = torch.Tensor(w_lag)

    return x_t,x_lag,w_t,w_lag

def divide_in_batches(list_of_tensors,batch_size):
    '''
    Takes as input a list of torch.tensors and returns batches of lenght batch_size
    
    Parameters:
        list_of_tensors (list of tensors)
        batch_size 
        
    Returns:
        list of batches
    '''
    batch=[]
    for x in list_of_tensors:
        batch.append(torch.split(x,batch_size,dim=0))
    
    batch=list(zip(*batch))
    
    return batch

def split_train_valid(x,t,n_train,n_valid,every,last_valid=False):
    print("[SPLIT DATASET]")
    x_train = x[:n_train*every:every]
    t_train = t[:n_train*every:every]
    print("- Training points =",x_train.shape[0])

    if last_valid:
        x_valid = x[-n_valid*every::every]
        t_valid = t[-n_valid*every::every]
    else:
        x_valid = x[n_train*every:(n_train+n_valid)*every:every]
        t_valid = t[n_train*every:(n_train+n_valid)*every:every]
    
    train = [x_train,t_train]
    valid = [x_valid,t_valid]
    
    return train,valid
    
def create_tica_dataset(x,t,lag_time,n_train,n_valid,every=1,batch_tr=-1,last_valid=False):
    '''Returns [x_t, x_lag, w_t, w_lag]''' 
    
    train,valid = split_train_valid(x,t,n_train,n_valid,every,last_valid=last_valid)
    x_train,t_train = train
    x_valid,t_valid = valid

    # TRAINING SET --> DATASET WITH PAIRS (x_t,x_t+lag)
    print("- Search (x_t,x_t+lag) with lag time =",lag_time)
    train_configs = look_for_configurations(x_train,t_train,lag=lag_time)
    print("- Found n_pairs =",train_configs[0].shape[0])

    # create batches
    if batch_tr == -1:
        batch_tr = len(train_configs[0])
    train_batches = divide_in_batches(train_configs,batch_tr)
    print("- Batch size =",batch_tr)
    print("- N batches =",len(train_batches))

    # VALID CONFIGS
    valid_configs = look_for_configurations(x_valid,t_valid,lag=lag_time)

    return train_batches,valid_configs