Neural Networks are at the core of deep learning. But these are often constrained by Back-propagation algorithm which requires the derivative of Loss function with respect to network parameters. In this repository, I will show that Neural Networks are not limited by back-propagation and we can use Simultaneous Perturbation using Stochastic Approximation(SPSA) to find noisy gradients. This technique is highly useful when it is very expensive to compute the gradients of the loss function or it is not differentiable at all. Gradient Descent or any other popular optimisation algorithms like Adam/RMSProp requires to compute Gradient.

## Usage

```bash
python3 train.py
```

You can read the full article at [http://anshulyadav.me/2019/06/21/SPSA-Neural-Network/](http://anshulyadav.me/2019/06/21/SPSA-Neural-Network/)
