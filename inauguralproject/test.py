def square(x):
    """ square numpy array
    
    Args:
    
        x (ndarray): input array
        
    Returns:
    
        y (ndarray): output array
    
    """
    
    y = x**2
    return y




class ExchangeClass:

    def __init__(self):
        """ setup """

        # a. parameters
        self.alpha = 1/3 # consumers A share of income used on good 1
        self.beta = 2/3 # consumers B share of income used on good 1
        self.endowment_1_A = 0.8
        self.endowment_2_A = 0.3
        self.endowment_1_B = 1-0.8
        self.endowment_2_B = 1-0.3
        
    def utility(self):
        """ Utility function """

        return (x1**self.alpha)*x2
    self.K**self.alpha*self.L**(1-self.alpha)

    def cost(self):
        """ cost """
        
        return self.rk*self.K + self.w*self.L

    def profit_bad(self):
        """ calculate profits """
        
        revenue = self.p*self.K**self.alpha*self.L**(1-self.alpha)
        cost = self.rk*self.K + self.w*self.L
        return revenue-cost

    def profit_good(self):
        """ calculate profits """

        revenue = self.p*self.production()
        cost = self.cost()
        return revenue-cost
