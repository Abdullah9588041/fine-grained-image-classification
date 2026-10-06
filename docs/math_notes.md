# Mathematical Notes — Fine-Grained Image Classification

## 1. The convolution operation

For input $X \in \mathbb{R}^{H \times W \times C_{in}}$ and kernel
$K \in \mathbb{R}^{k \times k \times C_{in} \times C_{out}}$, the (cross-correlation)
"convolution" used in CNNs is

$$(X \star K)_{i,j,o} = \sum_{m=1}^{k}\sum_{n=1}^{k}\sum_{c=1}^{C_{in}}
   X_{i+m-1,\,j+n-1,\,c}\; K_{m,n,c,o}.$$

Weight sharing (same $K$ at every location) gives translation equivariance:
shifting the input shifts the feature map identically. This is why a petal-edge
detector fires wherever a petal edge appears.

## 2. Softmax + cross-entropy and its gradient

Logits $z \in \mathbb{R}^{C}$, softmax $p_i = e^{z_i}/\sum_j e^{z_j}$,
cross-entropy against one-hot target $y$:

$$L = -\sum_i y_i \log p_i = -z_t + \log\sum_j e^{z_j}, \qquad t = \text{true class}.$$

Differentiating,

$$\frac{\partial L}{\partial z_i} = p_i - y_i.$$

This beautifully simple gradient is what backpropagates: the network is pushed to
raise the true-class logit and suppress the rest, in proportion to how wrong the
current probabilities are.

## 3. AdamW update rule

Adam maintains biased moment estimates $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$,
$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$, bias-corrects them
($\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$), and updates

$$\theta_t = \theta_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}
   + \lambda\,\theta_{t-1}\right).$$

The crucial "W" (Loshchilov & Hutter, 2019): weight decay $\lambda\theta$ is applied
*outside* the adaptive scaling, i.e. true $L_2$ regularization. In vanilla Adam the
decay term is folded into $g_t$ and gets adaptively rescaled, which weakens
regularization for frequently-updated parameters — exactly the wrong behavior when
fine-tuning a large pretrained backbone.

## 4. Grad-CAM (Selvaraju et al., ICCV 2017)

Let $A^k \in \mathbb{R}^{u \times v}$ be the $k$-th feature map of a target
convolutional layer and $y^c$ the pre-softmax score of class $c$. The importance
weight of map $k$ for class $c$ is the global-average-pooled gradient

$$\alpha_k^c = \frac{1}{uv}\sum_{i=1}^{u}\sum_{j=1}^{v}
   \frac{\partial y^c}{\partial A^k_{ij}},$$

and the class-discriminative localization map is

$$L^c = \mathrm{ReLU}\!\left(\sum_k \alpha_k^c A^k\right).$$

Intuition: $\partial y^c / \partial A^k_{ij}$ measures how much a small increase in
activation at spatial location $(i,j)$ would increase the class score; averaging
over space gives one weight per channel; the weighted combination highlights the
regions the network "looked at". ReLU keeps only positive influences on the class
of interest. We implement this from scratch with forward/backward hooks in
`src/flowers/explain.py` and target the last ResNet-50 block (`layer4`), whose
$7\times7$ maps carry the richest semantics before global pooling.

## 5. Why fine-grained classification is hard

With 102 visually similar species and only 10 training images per class, the
intra-class variance (lighting, pose, growth stage) can exceed the inter-class
variance (two daisy species differing only in petal count). Formally, the Bayes
error is high: class-conditional densities $p(x \mid c)$ overlap heavily in pixel
space. Transfer learning works because ImageNet-pretrained filters already span a
basis of texture/shape features in which the classes are *more* linearly separable
— the new head only needs to learn the separating hyperplane in that basis.

## References

- Nilsback & Zisserman (2008). Automated flower classification over a large number of classes.
- He et al. (2016). Deep residual learning for image recognition.
- Loshchilov & Hutter (2019). Decoupled weight decay regularization.
- Selvaraju et al. (2017). Grad-CAM: Visual explanations from deep networks.
