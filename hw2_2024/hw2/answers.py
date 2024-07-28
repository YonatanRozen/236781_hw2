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
first lets explain about dropout:
dropout is a regularization technique used to prevent overfitting during the training of neural networks.It works by randomly "dropping out" a fraction of the neurons in the network on each forward pass. This means that each neuron has some probability of being dropped. By randomly dropping neurons, dropout forces the network to not rely on specific neurons, thus preventing co-adaptation. This makes the network more robust and able to generalize better to unseen data.

1 + 2. lets first explain the graphs with dropout and without dropout. in the graph that does not use dropout we can see a very good training accuracy (more than 95% in just 30 epochs) but a consistent and relativly low test accuracy after 5 epochs. this is what we expected to see because as we explained above, when not using dropout the neural network relys more on spesific neurons and tends to overfit. this overfitting explains the low test accuracy despite high training accuracy. 
 on the other hand, we can see that the graph with dropout = 0.4 has lower training accuracy but much better test accuracy. again, this is what we expected to see because the dropout generalzed the network such that it does not rely on spesific neurons.
 in the last gragh with dropout = 0.8 we can see that the training accuracy is the lowest and test accuracy is similar to dropout = 0. we expected this because when so many neurons are dropd, the network loses important information and it learns less. it havily generalizes the data so the test accuracy is similar to the first graph.
 essentially, the dropout parameter is a trade-off between fitting to specific neurons and data, and generalization and robustness of the network.
 
"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible for the test loss to increase for a few epochs while the test accuracy also increases. the Cross-Entropy Loss measures the difference between the predicted probability distribution and the actual distribution. It takes into account the confidence of the predictions. If a model becomes more confident in its wrong predictions or less confident in its correct predictions, the loss can increase. 
the test accuracy measures the percentage of correct predictions. It does not take into account the confidence of the predictions, only whether the top predicted class matches the actual class. 
let us look at an examle: let there be $y'_1 = \hat{y'}_1$  and $y'_2 \ne \hat{y'}_2, y'_3 \ne \hat{y'}_3$ which are almost identical. Then, in the next epoch, $\hat{y'}_2, \hat{y'}_3$ increase by a small margin, such that $y'_2 = \hat{y'}_2, y'_3 = \hat{y'}_3$ and $y_1$ decreases by a big margin.
after this epoch, the predicted labels have not changed except for $y'_2$ and $y'_3$ that are now equal to their true labels and one predicted lable $y'_1$ that might be not equal to its true lable. therefore the test accuracy increases.
Despite that, the distances between the predicted and the true labels are greater than the last epoch and thus the loss also increses.

"""

part2_q3 = r"""
**Your answer:**
1.Gradient Descent is an optimization algorithm. this algorithm updates the model parameters in each iteration to minimize the loss function.
Back-Propagation is an algorithm used to compute the gradient of the loss function with respect to the model parameters in neural networks. It efficiently calculates the gradient using the chain rule.

2. SGD algorithm is similar to the GD algorithm in almost every aspect. the difference between them is that in SGD, in each epoch the algorithm samples a subset of the dataset $X$, which its size is $BatchSize$, and only take that subset into consideration whlie deciding the step. In GD the algorithm takes into consideration all the samples.
now we will compare the effects of this diffrence. in terms of speed and efficiency, GD is Slower per update because it processes the entire dataset but more stable due to averaging over all examples. SGD is Faster per update since it processes only one example at a time, but less stable due to high variance in parameter updates. 
in terms of convergence, GD Moves steadily towards the minimum but can be slow and may get stuck in local minima. SGD has Faster convergence due to frequent updates, but can fluctuate around the minimum and requires a learning rate schedule to stabilize.
in terms of memory usege, GD Requires the entire dataset in memory. SGD Requires only one example in memory at a time, suitable for large datasets.

3. a. when dealing with large datasets, in GD algorithm the gradient is calculated for the entire dataset, which can be computationally expensive and slow. In contrast, SGD updates the model parameters using only one or a few training examples at a time (mini-batches), making the update process much faster and more feasible for large-scale data.
   b. The inherent noise in the parameter updates due to the random selection of training examples in SGD acts as a form of regularization. This can help prevent overfitting and allow the model to generalize better to unseen data. The noise can help the model escape local minima and potentially find better solutions.
   c. The frequent updates in SGD lead to faster initial progress in training. This means that even with the same computational resources, SGD can achieve a reasonable level of model accuracy much faster than GD, which can be crucial in practical settings where quick iterations and feedback are necessary.
   
4. a. The proposed method would not produce a gradient strictly equivalent to that of standard GD due to the difference in scaling. In standard GD, the gradient is the average of the loss gradients across the entire dataset. In the proposed method, the gradient is the sum of the loss gradients from disjoint batches, making it potentially larger by a factor of the number of batches K. This discrepancy can lead to differences in update magnitudes and convergence behavior. Moreover, the balance of contributions from each sample in standard GD would not be maintained.
   b. When performing the backward pass in neural network training, it requires the memory to store the intermediate activations and gradients for all parameters across all samples that were part of the forward pass. When multiple batches are processed in forward passes before a single backward pass, the memory usage can spike dramatically. This is because the intermediate activations from each batch need to be retained until the backward pass is performed. Even though each individual batch might fit into memory during its forward pass, the combined memory usage from storing intermediate states for all batches can exceed the available memory. This accumulation leads to an out-of-memory error, despite using smaller batch sizes.
"""

part2_q4 = r"""
**Your answer:**
1.a. In forward mode AD, to compute the gradient ∇f(x0) of the function f=fn∘fn−1∘…∘f1 at some point x0, we track the function values and their derivatives during the forward pass. By storing only the current function value and its derivative at each step, we achieve O(1) memory complexity. This approach is efficient and avoids the need to store all intermediate values and their derivatives simultaneously (which causes the memory complexity to be O(n)), while maintaining the computation cost of O(n).
  b. To compute the gradient using backward mode Automatic Differentiation (AD) with reduced memory complexity, we can employ checkpointing. This method strategically stores a subset of intermediate values (checkpoints) instead of all intermediate values during the forward pass, and recomputes the necessary values on-the-fly during the backward pass. By dividing the computational graph into segments and storing values only at the beginning of each segment, checkpointing significantly reduces memory usage. During the backward pass, starting from the last checkpoint, we recompute intermediate values within each segment as needed to propagate the gradients backward. This allows us to manage memory efficiently, reducing the memory complexity to O(k) where k is the number of checkpoints, while maintaining the overall computation cost at O(n). if we choose to use 2 checkpoints for example, the memory complexity is O(1).
  
2. Yes, checkpointing techniques can be generalized for arbitrary computational graphs, though with some considerations. Checkpointing works by saving and recomputing intermediate values during the backward pass, allowing for significant memory savings. To apply this effectively, the graph can be decomposed into manageable subgraphs or segments, which can be processed sequentially to handle memory efficiently. However, for graphs with parallel executions or complex dependencies, checkpointing must be adapted to manage synchronization and dependencies. While achieving O(1) memory complexity in all cases may be challenging, checkpointing still offers substantial improvements over storing all intermediate activations, making it a valuable approach for handling large and complex computational graphs.   

3. in deep architectures like VGGs and ResNets, Applying checkpointing techniques to backpropagation significantly improves memory management. By saving intermediate activations only at selected checkpoints and recomputing them as needed during the backward pass, checkpointing reduces the memory footprint compared to storing all activations. Although the memory complexity does not reach O(1) (constant space) but rather O(k), where k is the number of checkpoints, this approach still substantially alleviates the memory burden. This reduction allows for training very deep networks on hardware with limited memory, making it feasible to handle large models.
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
1. The model's performance, as evidenced by the results, was quite poor. It frequently identified incorrect objects with high confidence. For instance, it mistakenly identified a dolphin as a person with a confidence level of 0.9, which is notably high. Accurate predictions were rare, with the model often marking non-existent items such as a surfboard and confusing different objects like dogs and cats.

2. In the first picture featuring dolphins, the misclassification may be due to the dolphins' dark coloration against a bright background, which poses a challenge for accurate detection. The dolphins overlapping each other adds to the difficulty, making it harder for the model to distinguish individual animals. Additionally, the model may have a bias towards detecting people, causing it to incorrectly classify dolphin shapes as humans. In the second picture, the main problem is the significant occlusion of the animals, which complicates the model's ability to accurately identify and differentiate their bodies. This occlusion likely leads to confusion in recognizing distinct animals. Moreover, the model's inherent bias towards certain objects might also play a role, as it struggles to correctly identify animals that are less commonly seen in occluded conditions.

3. To conduct a PGD attack on an object detection model like YOLO, begin with an initial image 𝑥 and utilize a pretrained YOLO model. The aim is to cause YOLO to either incorrectly classify the objects in 𝑥 or completely miss them. This is achieved by maximizing the model's loss through iterative perturbations added to the image. In each iteration, calculate the gradient of the loss with respect to the image and modify the image in the direction that increases the loss. It's crucial to keep these perturbations minimal to ensure the altered image remains similar to the original. By projecting the perturbations within a certain bound, you avoid making the image significantly different from its original version. This method subtly changes the image to mislead the model while keeping the modifications undetectable to the human eye.
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
In the first image, a range of animals including lions, tigers, and cheetahs are visible, all facing the camera. Unfortunately, the model struggled to accurately identify these animals, categorizing them all as either cats or birds. This issue likely arises from the model's training data, which predominantly features cats, leading to a bias towards recognizing feline characteristics. Additionally, partial occlusions of the animals in the image contributed to incorrect classifications, with the model misinterpreting the occluded animals as a single large creature. The occlusion even led the model to erroneously identify a bird that isn't present in the image. Despite the presence of distinct species like lions and tigers, the model’s narrow training scope caused it to generalize all animals as cats.

In the second image, two wolves are positioned very closely together, with their faces nearly touching against a white background. The model struggled to differentiate between them, mistakenly identifying them as a single large dog. This misclassification can be attributed to a few key factors. The overlap between the wolves created occlusion, leading to confusion in distinguishing the two animals. Additionally, the white background accentuated the wolves' dark fur, which may have caused the model to perceive them as a single entity. This situation underscores the need for diverse and thorough training datasets to enhance the model's capability to differentiate between objects that are closely spaced.

In the third image, a street scene is captured with significant motion blur as a car passes by. The model incorrectly identified the car as a train, which can be attributed to the blur and movement obscuring key details. The lack of clear, distinct features due to the blurriness likely hindered the model's ability to accurately classify the object. This situation highlights the difficulties models encounter with low-resolution or dynamic images and emphasizes the necessity for training on a variety of visual conditions to enhance the model's robustness.
"""

part6_bonus = r"""
**Your answer:**
To enhance the pictures, we convert them from their original color space to one more suited for processing. Next, we significantly boost the contrast to make the details more distinct and noticeble. After adjusting the contrast, we ensure that the brightness of images remains at their original level to keep their natural appearance intact. We then apply a slight blur to reduce noise and smooth out minor imperfections. Following this, we sharpen the images to enhance the edges and make the details more pronounced. Converting the images to grayscale offers a different perspective, emphasizing contrast and structure. Lastly, we resize the images for uniformity and save the new pictures. these manipulations on the pictures will help the model to classify objects better. for example we ran the code on the third picture and after that the model was able to classify it correctly.

"""