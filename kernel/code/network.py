import torch
import torch.nn as nn
import numpy as np

"""
This script implements a kernel logistic regression model, a radial basis function network model
and a two-layer feed forward network.
"""

class Kernel_Layer(nn.Module):

    def __init__(self, sigma, hidden_dim=None):
        """
        Set hyper-parameters.
        Args:
            sigma: the sigma for Gaussian kernel (radial basis function)
            hidden_dim: the number of "kernel units", default is None, then the number of "kernel units"
                                       will be set to be the number of training samples
        """
        super(Kernel_Layer, self).__init__()
        self.sigma = sigma
        self.hidden_dim = hidden_dim

    def reset_parameters(self, X):
        """
        Set prototypes (stored training samples or "representatives" of training samples) of
        the kernel layer.
        """
        if self.hidden_dim is not None:
            X = self._k_means(X)
        self.prototypes = nn.Parameter(torch.tensor(X).float(), requires_grad=False)

    def _k_means(self, X):
        """
        K-means clustering

        Args:
            X: A Numpy array of shape [n_samples, n_features].

        Returns:
            centroids: A Numpy array of shape [self.hidden_dim, n_features].
        """
        ### YOUR CODE HERE
        #centroids = torch.from_numpy(np.zeros((self.hidden_state, X.shape[1])))

        random_indices = torch.randperm(X.shape[0])[:self.hidden_dim]
        centroids = X[random_indices]

        classifications = dict()

        optimized = False
        while optimized is True:
            for i in range(self.hidden_dim):
                classifications.update({i:list()})

            for i in range(X.shape[0]):
                distances = [torch.linalg.norm(torch.tensor(X[i]-centroids[j])) for j in range(centroids.shape[0])]
                class_i = distances.index(min(distances))
                classifications[class_i].append(X[i])

            prev_centroids = centroids
            for class_i in classifications:
                #numpy_arr = [x.numpy() for x in classifications[class_i]]
                if(len(classifications[class_i]) > 0): centroids[class_i] = torch.from_numpy(np.average(classifications[class_i],axis=0))
                else: centroids[class_i] = 0
            for i in range(len(centroids)):
                if((int((centroids[i]-prev_centroids[i]).sum(0))) == 0):
                    optimized = True
                    break

        ### END YOUR CODE
        return centroids

    def forward(self, x):
        """
        Compute Gaussian kernel (radial basis function) of the input sample batch
        and self.prototypes (stored training samples or "representatives" of training samples).

        Args:
            x: A torch tensor of shape [batch_size, n_features]

        Returns:
            A torch tensor of shape [batch_size, num_of_prototypes]
        """
        assert x.shape[1] == self.prototypes.shape[1]
        ### YOUR CODE HERE
        # Basically you need to follow the equation of radial basis function
        # in the section 5 of note at http://people.tamu.edu/~sji/classes/nnkernel.pdf
        rbf_kernel = torch.from_numpy(np.random.rand(x.shape[0],self.prototypes.shape[0]))
        rbf_kernel = rbf_kernel.cuda()
        sigma = torch.tensor(self.sigma)

        for i, x in enumerate(x.cpu().numpy()):
            for j, y in enumerate(self.prototypes.cpu().numpy()):
                sum = 0.0
                for k in range(self.prototypes.shape[1]):
                    sum += (x[k] - y[k])**2
                rbf_kernel[i][j] = torch.exp(-torch.tensor(sum)/(2*sigma**2))

        return(rbf_kernel.float())
        ### END YOUR CODE


class Kernel_LR(nn.Module):

    def __init__(self, sigma, hidden_dim):
        """
        Define network structure.

        Args:
            sigma: used in the kernel layer.
            hidden_dim: the number of prototypes in the kernel layer,
                                       in this model, hidden_dim has to be equal to the
                                       number of training samples.
        """
        super(Kernel_LR, self).__init__()
        self.hidden_dim = hidden_dim
        ### YOUR CODE HERE
        # Use pytorch nn.Sequential object to build a network composed of a
        # kernel layer (Kernel_Layer object) and a linear layer (nn.Linear object)

        # Remember that kernel logistic regression model uses all training samples
        # in kernel layer, so set 'hidden_dim' argument to be None when creating
        # a Kernel_Layer object.

        # How should we set the "bias" argument of nn.Linear?
        self.net = nn.Sequential(
            Kernel_Layer(sigma, None),
            nn.Linear(hidden_dim,1)
        )

        ### END YOUR CODE

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: A torch tensor of shape [batch_size, n_features]

        Returns:
            A torch tensor of shape [batch_size, 1]
        """
        return self.net(x)

    def reset_parameters(self, X):
        """
        Initialize the weights of the linear layer and the prototypes of the kernel layer.

        Args:
            X: A Numpy array of shape [n_samples, n_features], training data matrix.
        """
        assert X.shape[0] == self.hidden_dim
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                if isinstance(layer, Kernel_Layer):
                    layer.reset_parameters(X)
                else:
                    layer.reset_parameters()



class RBF(nn.Module):

    def __init__(self, sigma, hidden_dim):
        """
        Define network structure.

        Args:
            sigma: used in the kernel layer.
            hidden_dim: the number of prototypes in the kernel layer,
                                       in this model, hidden_dim is a user-specified hyper-parameter.
        """
        super(RBF, self).__init__()
        ### YOUR CODE HERE
        # Use pytorch nn.Sequential object to build a network composed of a
        # kernel layer (Kernel_Layer object) and a linear layer (nn.Linear object)
        # How should we set the "bias" argument of nn.Linear?
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            Kernel_Layer(sigma, hidden_dim),
            nn.Linear(hidden_dim,1)
        )

        ### END CODE HERE

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: A torch tensor of shape [batch_size, n_features]

        Returns:
            A torch tensor of shape [batch_size, 1]
        """
        return self.net(x)

    def reset_parameters(self, X):
        """
        Initialize the weights of the linear layer and the prototypes of the kernel layer.

        Args:
            X: A Numpy array of shape [n_samples, n_features], training data matrix.
        """
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                if isinstance(layer, Kernel_Layer):
                    layer.reset_parameters(X)
                else:
                    layer.reset_parameters()



class FFN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        """
        Define network structure.

        Args:
            input_dim: number of features of each input.
            hidden_dim: the number of hidden units in the hidden layer, a user-specified hyper-parameter.
        """
        super(FFN, self).__init__()
        ### YOUR CODE HERE
        # Use pytorch nn.Sequential object to build a network composed of
        # two linear layers (nn.Linear object)
        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Linear(hidden_dim,1)
        )
        ### END CODE HERE

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: A torch tensor of shape [batch_size, n_features]

        Returns:
            A torch tensor of shape [batch_size, 1]
        """
        return self.net(x)

    def reset_parameters(self):
        """
        Initialize the weights of the linear layers.
        """
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
