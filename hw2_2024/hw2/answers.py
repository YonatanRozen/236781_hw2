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

1. Optimization error - refers to the difference between the model's predictions and the 
actual values on the training data. This error is typically related to the model's ability to fit the 
data and usually caused by bad initialization, optimization technique model architecture etc. 

In CNNS, this error can be reduced by improving the model's architecture 
and making sure the receptive field of each layer is sufficient for the model to learn
important patterns in the data. This can increase the ability of the model to fit the data, and lower
the optimization error. 


2. Generalization error - refers to the model's ability to correctly predict classes of examples which weren't 
included in the training (to "generalize"). This error typically happens when the model is too biased about 
the training data and over fits it, thus performing poorly on general data.

A high generalization error can be reduced with regularization which involves penalizing complex models to reduce the
 chance of over fitting, cross-validation which involves a pre-processing phase which picks hyperparameters based on 
 approximated generalization error, and adjusting the receptive field of convolution layers to prevent irrelevant
 features to be learned by them and cause over fitting.
 

3. Approximation error - quantifies how much the model is suitable for fitting the data. A high approximation
error typically means that the model is too simple to learn the relationships in the data. 

It can be lowered by increasing model complexity and using a more advanced architecture. In CNNs specifically,
we can expand the receptive field of each layer, add more layers, use more advanced architectures like residual 
blocks and etc. 

Generally, lowering each of these errors will consequently lower the population error. 

"""

part3_q2 = r"""
**Your answer:**

A higher FPR makes more sense in cases in which classifying an example as positive by mistake (FP), causes less
damage than missing an actually positive example (FN). A good example for this is a database of medical tests
for corona virus for example. We prefer to diagnose someone as sick by mistake, than to diagnose a sick person
as healthy and risking mass infection. 


"""

part3_q3 = r"""
**Im assuming that "optimal point on the ROC curve" means a point where the threshold values for the classifier is optimal
in terms ov FNR and FPR**

1. If the disease causes symptoms which aren't lethal in early phases, our focus will be to reduce the FPR
thus lowering the cost of unnecessary tests for patients who aren't sick. Later on, we can diagnose the disease
on the false negatives when symptoms begin to appear, making sure they don't lose their lives. 

2. If a person with the disease doesn't show any clear symptoms and may die in high probability if not 
diagnosed, we prefer saving lives. This means we want a lower FNR, and accept a higher FPR in the goal
of saving more lives and diagnose more people early. 

"""


part3_q4 = r"""
**Your answer:**

The problem with using MLPs on sequential data is that MLPs treat each input they get independently and they 
don't consider the context of the input and its relationship with other inputs. This makes them hard to use
when trying to learn the sentiment of a sentence, which is a task that requires learning the context of each word
in addition to the word itself. For example, say we want to learn the meaning of "it's not rainy today" 
with an MLP. The MLP doesn't capture the effect that "not" has on the following word "rainy", which is negation, so 
it might predict a wrong sentiment for this sentence. Also, MLPs require an input which is fixed in length, 
and since the length of sentences and words vary, it will be hard to use a fixed input length for this task. We
can break the sentence to fixed size parts, but this can result in fixed size parts which have no meaning 
and contain half words, which changes the sentiment of the sentence.  

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

    1. As we can see in the graphs, the test accuracy was lower as the depth increased and the depth that 
    produced the best results was 2. For example, we can see that when L=4 we get
    higher training accuracy than when L=2, but lower test accuracy. 
    This can be caused by over fitting: deeper networks are more expressive
    and learn more complex patterns in the data, however, they are more prone to learn irrelevant patterns and 
    noise which can lead to poor generalization. 
    
    2. The network wasn't trainable for K=8,16. We assume the reason for this is the phenomenon of vanishing gradients,
    which typically occurs when the number of layers is high. The high amount of layers causes gradients from the 
    higher layers to "disappear" when performing backward propagation, turning the final gradients eventually to zero. 
    This hurts th model's ability to learn and update its parameters based on the gradients. This problem can be
    partially solved with incorporation of skip layers such as residual blocks. These can allow gradients to flow
    directly to lower layers, without them vanishing and turning the whole gradient to zero.
    
"""

part5_q2 = r"""

    Examination of the graphs brings up a few conclusions: 
    1. For a fixed value of L=2 or L=4, bigger K values achieve higher training and test accuracies. This increase
        in accuracy is much more clear than in experiment 1_1. 
    2. For L=8 all values of K achieve very low accuracies. This supports our previous assumption in question 1.2.
    3. The configuration that achieved the best results is L=2, K=128, in comparison to experiment 1_1 where the 
    best configuration was L=2, K=64. 
    So overall, we see a clear increase in accuracy as K grows (for L=2,4) and the best configuration was different than
    in experiment 1_1.

"""

part5_q3 = r"""

    When examining the graphs, we can see that the model's performance varies for different depths. The depth
    that achieved the highest accuracy is a moderate depth L=3, in contrast to previous experiments. Also, a depth of 2 
    achieved higher train accuracy than a depth of 3 in almost all epochs, but lower test accuracy. This indicates 
    that a depth of 3 enables the model to learn more complex patterns in the data and generalize better than with 
    depth 2. Finally, we can see that when L=4 the model isn't trainable. This aligns with our previous assumption 
    (since L=4 with K=[64,128] means there are are 8 conv layers) and could indicate the presence of vanishing gradients. 
    
"""

part5_q4 = r"""

First of all, as in experiment 1.1 we can see that lower depths produced better test accuracies. This 
only partially aligns with experiment 1.3 since the best depth in experiment 1.3 was moderate. Another 
important observation is that when K is fixed to 32, the problem of vanishing gradients in higher depths
is gone. This happens thanks to the residual blocks in resnet, which can help with this problem as we explained 
in q1.2. Moreover, the vanishing gradients problem was dealt with success even when K was increased to 
bigger sizes ([64,128,256]), and L=2 achieved the best test accuracies.   

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