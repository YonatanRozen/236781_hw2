r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1. 
    a. The shape of the tensor is (64 , 512, 64, 1024). The jacobian of Y w.r.t X contains the derivatives of each 
    entry of Y w.r.t to each entry of X. This means that each entry of Y gives us a matrix of partial derivatives 
    which is the same shape as X. This is similar to when we calculated the derivative of cross entropy loss according 
    to X and the result was of the same shape as X. Since Y is a matrix of shape 64x512, and each entry is 
    translated to a matrix of shape 64x1024 (same size as X), we get a matrix of shape (64, 512, 64, 1024) which is a
    matrix of matrices. 
    
    b. The Jacobian is sparse. As we explained earlier, we differentiate each entry of y according to each entry in
        X. But each entry in Y is related to a specific sample (in fact, the entire row of that entry), so
         differentiating it according to any of the other 63 samples will produce 0. Thus, we have 63*1024 entries
        which are equal to 0, in every calculation in a specific cell in y, so we have 64*512*63*1024 0 entries in 
        the Jacobian which makes it sparse. 
        
    c. As written earlier in the notebook, we can calculate a specific entry in the gradient, 
    like \frac{\partial L}{\partial x_{1,1}} and than extrapolate from there how to obtain the VJP. 

2. 
    a. Similarly to what we explained in 1.a each entry in Y is differentiated w.r.t to each entry in W, 
       so the Jacobian dimensions will be (64, 512, 512, 1024).
       
    b. 

    c. As written earlier in the notebook, we can calculate a specific entry in the gradient, 
    like \frac{\partial L}{\partial w_{1,1}} and than extrapolate from there how to obtain the VJP.
        

"""

part1_q2 = r"""

**Your answer:**

    backpropagation is indeed required for descent optimization. Neural networks typically include multiple 
    layers, so computing gradients of a layer according to inputs of previous layers directly might be 
    computationally expensive. Using backpropagation allows propagation
    of gradients using partial calculations from higher layers, which turns this process to more efficient.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    reg = 0.0011
    lr = 0.051
    wstd = 0.11
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    lr_rmsprop = 0.00019
    reg = 0.00151
    lr_vanilla = 0.019
    lr_momentum = 0.0041
    wstd = 0.051
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    lr = 0.0011
    wstd = 0.151
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.09, 0.001, 0.3  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""

1. Residual block:
    For each convolutional layer, there are 256 filters of size 3x3x256 + 1 (+1 because of the bias)
    => we have total of ((3x3x256+1)x256)x2 = 1,180,160 parameters.
    
    Bottleneck residual block: 
    The 1st layer projects the input to 64 channels using 64 (1x1x256 + 1) filters => 257x64 = 16448 parameters.
    The 2nd layer has 64 (3x3x64 + 1) filters => (3x3x64 + 1)x64 = 36928 parameters.
    The 3rd layer expands the input back to 256 channels using 256 (1x1x64 + 1) filters => 256x64 + 256 = 16640 parameters.
    => in total we have 16448 + 36928 + 16640 = 70016 parameters.
    
2. As we see, the number of parameters and the shape of filters is significantly bigger in a regular residual block.
   The number of floating point operations needed for an output is directly affected by this, so we expect it 
   to be bigger in a regular residual block than in bottleneck.
    
3. (1). The regular residual block has two 3x3 convolution layers, in contrast to the bottleneck block which has 
   only one 3x3 convolution and two 1x1 convolutions. This gives the regular block more power in combining
   the input spatially and discover spatial relationships within feature maps. The bottleneck block on the 
   hand has a lower receptive field.
   
   (2). The regular residual block has more power in combining the input across feature maps, since
    it doesn't reduce the number of channels. It has more filters and so a wider variety of ways combine 
    the channels. The bottleneck block, on the other hand, reduces the number of channels the input has, 
    thus loosing some of the cross-channel information which was possible to capture beforehand.   

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""