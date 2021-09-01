import torch

from . import utils

##################################
# Base Neural Network Class
##################################

class BaseNNCV(torch.nn.Module):

    def __init__(self):
        super().__init__()
    
    def standardize_inputs(self, x: torch.Tensor, print_values=False):
        '''
        Enable the standardization of inputs based on max and min over set.

        Parameters:
            x (tensor): reference set over which compute the standardization
        '''
        
        #compute mean and range
        Max, _=torch.max(x,dim=0)
        Min, _=torch.min(x,dim=0)

        Mean=(Max+Min)/2.
        Range=(Max-Min)/2.

        if print_values:
            print( "Input standardization enabled.")
            print('Mean:',Mean.shape,'-->',Mean)
            print('Range:',Range.shape,'-->',Range)
        if (Range<1e-6).nonzero().sum() > 0 :
            print( "[Warninig] Input normalization: the following features have a range of values < 1e-6:", (Range<1e-6).nonzero() )
            Range[Range<1e-6]=1.

        self.normIn = True
        self.MeanIn = Mean
        self.RangeIn = Range

    def standardize_outputs(self, input: torch.Tensor, print_info=False):
        '''
        Enable the standardization of outputs based on max and min over set.

        Parameters:
            x (tensor): input set over which compute the standardization
        '''
        #disable normout for cv evaluation
        self.normOut = False
        with torch.no_grad():
            x = self.forward(input)

        #compute mean and range
        Max, _=torch.max(x,dim=0)
        Min, _=torch.min(x,dim=0)

        Mean=(Max+Min)/2.
        Range=(Max-Min)/2.

        if print_info:
            print( "Output standardization enabled.")
            print('Mean:',Mean.shape,'-->',Mean)
            print('Range:',Range.shape,'-->',Range)
        if (Range<1e-6).nonzero().sum() > 0 :
            print( "[Warninig] Output normalization: the following features have a range of values < 1e-6:", (Range<1e-6).nonzero() )
            Range[Range<1e-6]=1.

        self.normOut = True
        self.MeanOut = Mean
        self.RangeOut = Range

    def disable_standardize_outputs(self,print_info=False):
        '''
        Disable the standardization of outputs.
        '''
        if print_info:
            print( "Output standardization disabled.")
        self.normOut = False

    def _normalize(self, x: torch.Tensor, Mean: torch.Tensor, Range: torch.Tensor) -> (torch.Tensor):
        ''' Compute standardized outputs '''

        batch_size = x.size(0)
        x_size = x.size(1)

        Mean_ = Mean.unsqueeze(0).expand(batch_size, x_size)
        Range_ = Range.unsqueeze(0).expand(batch_size, x_size)

        return x.sub(Mean_).div(Range_)
    
    def set_optimizer(self,opt):
        self.opt_ = opt
        
    def default_optimizer(self):
        self.opt_ = torch.optim.Adam(self.parameters())
        
    def set_earlystopping(self,patience=5, min_delta = 0, consecutive=True, log=False, save_best_model=True):
        self.earlystopping_ = utils.EarlyStopping(patience,min_delta,consecutive,log,save_best_model)
        self.best_valid = None
        self.best_model = None
        
    def set_lrscheduler(self,):
        self.lrscheduler_ = utils.LRScheduler(opt)
        
    def print_info(self):
        print('================INFO================')
        print('[MODEL]')
        print(self)
        print('\n[OPTIMIZER]')
        print(self.opt_)
        print('\n[PARAMETERS]')
        print(self.get_params())
        print('====================================')
        
    def print_log(self,log_dict,spacing=None,decimals=3):
        if spacing is None:
            spacing = [16 for i in range(len(log_dict))]
        if self.log_header:
            for i,key in enumerate(log_dict.keys()):
                print("{0:<{width}s}".format(key,width=spacing[i]),end='')
            print('')
            self.log_header = False
            
        for i,value in enumerate(log_dict.values()):
            if type(value) == int:
                print("{0:<{width}d}".format(value,width=spacing[i]),end='')
                
            if (type(value) == torch.Tensor) or (type(value) == torch.nn.parameter.Parameter) :
                value = value.numpy()
                if value.shape == ():
                    print("{0:<{width}.{dec}f}".format(value,width=spacing[i],dec=decimals),end='')
                else:
                    for v in value:
                        print("{0:<6.3f}".format(v),end=' ')
        print('')
