import torch
import typing


#LDS:Algo 2 Step #8 and #12 from LDS paper 
#Bernoulli random variable sampling:  This is based on the basic idea of sampling uniform 
#rv and if that rv  < theta then the bern rv is 1 otherwise zero. This is worked out in 
#the sample function. The sampler function returns a lambda function, which takes theta 
#as input and gives bern rv as output. 

#Question: Why to we need ndim? shape here is an int and int doesn't have a length.
# So, ndim should fail 
class BernoulliRV:

    def __init__(self, shape):
        self.shape = shape

    def _sample(self, theta):
        """
        Draws one sample of the rv with success parameter theta

        :param theta: the success parameter in [0, 1]
        :return: a sample
        """
        uni = torch.rand(self.shape)  # , generator=generator)
        return (torch.sign(theta - uni) + 1) / 2

    def sampler(self):
        """
        :return: a sampler for the rv (a callable)
        """
        return lambda theta: self._sample(theta)

    @property
    def ndim(self):
        """
        :return: dimensionality of this rv
        """
        return len(self.shape)
        
#------------------------------------------------------------------------------------------        
#ste function: takes in sampler and returns a lambda function(which is a callable)
#the lambda function takes in sampler and theta (parameter for bernoulli) and pass those
#to _StraightThroughEstimator.apply(). The returned callable implements forward and backward function

def ste(sampler: typing.Callable):
    """
    Function to call for initializing a straight-through estimator

    :param sampler: a function that returns a sample of the random variable
    :return: a callable that implements the forward and the backward pass
    """
    return lambda theta: _StraightThroughEstimator.apply(theta, sampler)


#It is a subclass of a torch.autograd.Function. We can extend autograd with this 
#subclassing to implement our own custom autograd. 
#forward function does forward pass i.e. just samples bern rv 
#backward function does backprop. The gradient of the function 
# For STE, we want grad_output

class _StraightThroughEstimator(torch.autograd.Function):

    @staticmethod
    #computes output tensors from input tensors
    # forward returns 0, 1 suing theta as parameter for Bern
    def forward(ctx, theta, sampler):
        return sampler(theta)

    @staticmethod
    #we get tensor containing the gradient of loss 
    #w.r.t the output (grad_output)and want to caculate gradient for loss 
    #w.r.t input. STE approximates gradient of loss w.r.t. the input as
    #gradient of loss w.r.t. the output
    def backward(ctx, grad_outputs):
        return grad_outputs, None

#------------------------------------------------------------------------------------------

#By extending torch.nn.Module, we can specify our own neural net module.
# Question: forward function isn't going to work here right? The reason for that is the 
#ndim
class BernoulliSTEOp(torch.nn.Module):

    def __init__(self, shape):
        """
        In the constructor we have instantiated dist and ste_layer.
        dist is an instance of BernoulliRV class
        ste_layer is callable (returned by ste function)
        """
        super().__init__()
        self.dist = BernoulliRV(shape)
        self.ste_layer = ste(self.dist.sampler())

    def forward(self, theta):  # bs = batch size
    # I didn't understand this
        if theta.ndim == self.dist.ndim:  # no batch size, e.g. this is a `root' layer
            return self.ste_layer(theta)
        else:
            return torch.stack([self.forward(t) for t in theta])  # parallelization?



class BernoulliSTERoot(torch.nn.Module):

    def __init__(self, shape, initial_value=None):
        super().__init__()
        
        self.op = BernoulliSTEOp(shape)
        self.theta = torch.nn.Parameter(torch.Tensor(*shape))
        self.initialize(initial_value)

    def initialize(self, value=None):
        if value is not None:
            self.theta.data = value
        else:
            torch.nn.init.uniform_(self.theta, 0., 1.)

    def forward(self):
        if self.training:
            return self.op.forward(self.theta)

        return self.theta > 0.5
        
#this ensures that theta remains between 0 and 1 ..see demo()

    def project(self):
        self.theta.data = self.theta.data.clamp(0., 1.)


def demo(n):
    # just a Bernoulli random variable (wrapped into a `root layer') that uses the straight-through estimator
    # for computing the gradient
    model = BernoulliSTERoot((n,))  # you can also pass an initial value here
    print(model.theta)  # initial value of the success parameter
    b = model().clone().detach()  # NOTE: it's essential to call `detach' here.
    # Otherwise the gradient remains zero. Not sure why though...

    mse_loss = torch.nn.MSELoss()

    lss = mse_loss(model(), b)
    print(lss)

    # check that the gradient is properly computed
    grad = torch.autograd.grad(lss, [model.theta])
    print(grad)

    # initialize optimizer
    opt = torch.optim.SGD(model.parameters(), 1.)

    for i in range(20):
        opt.zero_grad()
        lss = mse_loss(model(), b)
        lss.backward()
        print(lss, model.theta.grad, model.theta, sep='\n', end='\n' + '='*20+'\n')
        opt.step()

        model.project()  # this ensures that theta remains between 0 and 1

    print('END')
    print(b)
    print(model.theta)

    # try now with a similar setting but with a slightly more complex ``dummy model'' with
    # a Bernoulli ste root and then a dense layer

    # ste_root = BernoulliSTERoot((n,))
    # dense = torch.nn.Linear(n, n, bias=False)
    # model = torch.nn.Sequential(ste_root, dense)


if __name__ == '__main__':
    demo(10)