# Logistic_regression

Logistic regression from Scratch

The loss function is this one :

<a href="http://www.codecogs.com/eqnedit.php?latex=C=\frac{1}{N}\sum_{i=0}^{N-1}[y_{true}\cdot&space;log(y_{pred})&plus;&space;(1-y_{true})\cdot&space;log({1-y_{pred}})]" target="_blank"><img src="http://latex.codecogs.com/png.latex?C=\frac{1}{N}\sum_{i=0}^{N-1}[y_{true}\cdot&space;log(y_{pred})&plus;&space;(1-y_{true})\cdot&space;log({1-y_{pred}})]" title="C=\frac{1}{N}\sum_{i=0}^{N-1}[y_{true}\cdot log(y_{pred})+ (1-y_{true})\cdot log({1-y_{pred}})]" /></a>

<a href="http://www.codecogs.com/eqnedit.php?latex=y_{pred}=\frac{1}{1&plus;e^{-w\cdot&space;x}}&space;\forall&space;x&space;\in[0,N-1]" target="_blank"><img src="http://latex.codecogs.com/png.latex?y_{pred}=\frac{1}{1&plus;e^{-w\cdot&space;x}}&space;\forall&space;x&space;\in[0,N-1]" title="y_{pred}=\frac{1}{1+e^{-w\cdot x}} \forall x \in[0,N-1]" /></a>

And its derivative is :

<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;C}{\partial&space;w}=-\frac{1}{n}\sum_{i=0}^{N-1}[-x&space;&plus;&space;x\cdot&space;\frac{e^{-w\cdot&space;x}}{1&plus;e^{-w\cdot&space;x}}&plus;y_{true}\cdot&space;x]" target="_blank"><img src="http://latex.codecogs.com/png.latex?\frac{\partial&space;C}{\partial&space;w}=-\frac{1}{n}\sum_{i=0}^{N-1}[-x&space;&plus;&space;x\cdot&space;\frac{e^{-w\cdot&space;x}}{1&plus;e^{-w\cdot&space;x}}&plus;y_{true}\cdot&space;x]" title="\frac{\partial C}{\partial w}=-\frac{1}{n}\sum_{i=0}^{N-1}[-x + x\cdot \frac{e^{-w\cdot x}}{1+e^{-w\cdot x}}+y_{true}\cdot x]" /></a>


We minimize C by using the gradient descent (which by the way can be accelerated by using momentum or nesterov acceleration for instance).
