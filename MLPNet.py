import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import Input, Model

tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tf.keras.layers

    
class MLPNet(keras.Model):
    """Create a Multilayer Perceptron (MLP) model with Normal distribution output."""

    def __init__(self, num_of_layers=3, nodes=1024, seed=2345, name="MLPNet", **kwargs):
        """
        Initialize the MLPNet model.

        Args:
            num_of_layers (int): Number of layers in the MLP.
            nodes (int): Number of nodes in each layer.
            seed: Random seed.
            name (str): Name of the model (default is 'MLPNet').
        """
        super(MLPNet, self).__init__(name=name, **kwargs)

        self.mlpnet_layers = []  # List to hold layers
        self.num_of_layers = num_of_layers  # Number of layers in the network
        self.nodes = nodes  # Number of nodes in each layer
        activation = tfkl.PReLU(alpha_initializer=tf.initializers.constant(0.3))

        # Create layers for the MLP network
        for li in range(num_of_layers):
            self.mlpnet_layers.append(tfkl.Dense(nodes*2, name='dense' + str(li + 1)))
            self.mlpnet_layers.append(tfkl.BatchNormalization(name='bn' + str(li + 1)))
            self.mlpnet_layers.append(tfkl.Activation(activation, name='act' + str(li + 1)))

        # Define the probability distribution layer (Normal distribution with mean and scale)
        self.prob = tfpl.DistributionLambda(lambda t: tfd.Normal(
            loc=t[..., :nodes], scale=1e-5 + tf.math.softplus(t[..., nodes:])), name='dist')

    
    def summary(self, pixels):
        
        """
        Generate a summary of the model architecture.
    
        Args:
            pixels (tuple): Shape of the input layer.
    
        Returns:
            Summary of the model architecture.
        """
        
        x = Input(shape=pixels)
        model = Model(inputs=x, outputs=self.call(x))
        return model.summary()
    
    
    def call(self, inputs):
        """
        Forward pass through the MLPNet model.

        Args:
            inputs: Input tensor.

        Returns:
            Probability distribution output.
        """
        x = inputs

        # Pass input through the defined layers
        for li, layer in enumerate(self.mlpnet_layers.layers):
            layer._name = 'MLPLayer' + str(li + 1)
            x = layer(x)

        x = self.prob(x)  # Apply probability distribution to the output x
        return x

#model = MLPNet(3, 1024,12345)
#model.summary(1024)