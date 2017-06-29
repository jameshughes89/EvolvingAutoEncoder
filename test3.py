import numpy as np
import matplotlib.pylab as plt
import gzip
import cPickle
import tensorflow as tf
import input_data_cifar

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=500):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, n_hidden_recog_3, n_hidden_recog_4, n_hidden_recog_5, n_hidden_recog_6, n_hidden_recog_7, n_hidden_recog_8, n_hidden_recog_9, n_hidden_recog_10, 
                            n_hidden_gener_1,  n_hidden_gener_2,  n_hidden_gener_3, n_hidden_gener_4, n_hidden_gener_5, n_hidden_gener_6, n_hidden_gener_7, n_hidden_gener_8, n_hidden_gener_9, n_hidden_gener_10,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'h3': tf.Variable(xavier_init(n_hidden_recog_2, n_hidden_recog_3)),
            'h4': tf.Variable(xavier_init(n_hidden_recog_3, n_hidden_recog_4)),
            'h5': tf.Variable(xavier_init(n_hidden_recog_4, n_hidden_recog_4)),
            'h6': tf.Variable(xavier_init(n_hidden_recog_5, n_hidden_recog_6)),
            'h7': tf.Variable(xavier_init(n_hidden_recog_6, n_hidden_recog_7)),
            'h8': tf.Variable(xavier_init(n_hidden_recog_7, n_hidden_recog_8)),
            'h9': tf.Variable(xavier_init(n_hidden_recog_8, n_hidden_recog_9)),
            'h10': tf.Variable(xavier_init(n_hidden_recog_9, n_hidden_recog_10)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_10, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_10, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'b3': tf.Variable(tf.zeros([n_hidden_recog_3], dtype=tf.float32)),
            'b4': tf.Variable(tf.zeros([n_hidden_recog_4], dtype=tf.float32)),
            'b5': tf.Variable(tf.zeros([n_hidden_recog_5], dtype=tf.float32)),
            'b6': tf.Variable(tf.zeros([n_hidden_recog_6], dtype=tf.float32)),
            'b7': tf.Variable(tf.zeros([n_hidden_recog_7], dtype=tf.float32)),
            'b8': tf.Variable(tf.zeros([n_hidden_recog_8], dtype=tf.float32)),
            'b9': tf.Variable(tf.zeros([n_hidden_recog_9], dtype=tf.float32)),
            'b10': tf.Variable(tf.zeros([n_hidden_recog_10], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'h3': tf.Variable(xavier_init(n_hidden_gener_2, n_hidden_gener_3)),
            'h4': tf.Variable(xavier_init(n_hidden_gener_3, n_hidden_gener_4)),
            'h5': tf.Variable(xavier_init(n_hidden_gener_4, n_hidden_gener_5)),
            'h6': tf.Variable(xavier_init(n_hidden_gener_5, n_hidden_gener_6)),
            'h7': tf.Variable(xavier_init(n_hidden_gener_6, n_hidden_gener_7)),
            'h8': tf.Variable(xavier_init(n_hidden_gener_7, n_hidden_gener_8)),
            'h9': tf.Variable(xavier_init(n_hidden_gener_8, n_hidden_gener_9)),
            'h10': tf.Variable(xavier_init(n_hidden_gener_9, n_hidden_gener_10)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_10, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_10, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'b3': tf.Variable(tf.zeros([n_hidden_gener_3], dtype=tf.float32)),
            'b4': tf.Variable(tf.zeros([n_hidden_gener_4], dtype=tf.float32)),
            'b5': tf.Variable(tf.zeros([n_hidden_gener_5], dtype=tf.float32)),
            'b6': tf.Variable(tf.zeros([n_hidden_gener_6], dtype=tf.float32)),
            'b7': tf.Variable(tf.zeros([n_hidden_gener_7], dtype=tf.float32)),
            'b8': tf.Variable(tf.zeros([n_hidden_gener_8], dtype=tf.float32)),
            'b9': tf.Variable(tf.zeros([n_hidden_gener_9], dtype=tf.float32)),
            'b10': tf.Variable(tf.zeros([n_hidden_gener_10], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        layer_3 = self.transfer_fct(tf.add(tf.matmul(layer_2, weights['h3']), 
                                           biases['b3'])) 
        layer_4 = self.transfer_fct(tf.add(tf.matmul(layer_3, weights['h4']), 
                                           biases['b4'])) 
        layer_5 = self.transfer_fct(tf.add(tf.matmul(layer_4, weights['h5']), 
                                           biases['b5'])) 
        layer_6 = self.transfer_fct(tf.add(tf.matmul(layer_5, weights['h6']), 
                                           biases['b6'])) 
        layer_7 = self.transfer_fct(tf.add(tf.matmul(layer_6, weights['h7']), 
                                           biases['b7'])) 
        layer_8 = self.transfer_fct(tf.add(tf.matmul(layer_7, weights['h8']), 
                                           biases['b8'])) 
        layer_9 = self.transfer_fct(tf.add(tf.matmul(layer_8, weights['h9']), 
                                           biases['b9'])) 
        layer_10 = self.transfer_fct(tf.add(tf.matmul(layer_9, weights['h10']), 
                                           biases['b10'])) 
        z_mean = tf.add(tf.matmul(layer_10, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_10, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        layer_3 = self.transfer_fct(tf.add(tf.matmul(layer_2, weights['h3']), 
                                           biases['b3'])) 
        layer_4 = self.transfer_fct(tf.add(tf.matmul(layer_3, weights['h4']), 
                                           biases['b4'])) 
        layer_5 = self.transfer_fct(tf.add(tf.matmul(layer_3, weights['h5']), 
                                           biases['b5'])) 
        layer_6 = self.transfer_fct(tf.add(tf.matmul(layer_3, weights['h6']), 
                                           biases['b6'])) 
        layer_7 = self.transfer_fct(tf.add(tf.matmul(layer_3, weights['h7']), 
                                           biases['b7'])) 
        layer_8 = self.transfer_fct(tf.add(tf.matmul(layer_3, weights['h8']), 
                                           biases['b8'])) 
        layer_9 = self.transfer_fct(tf.add(tf.matmul(layer_3, weights['h9']), 
                                           biases['b9'])) 
        layer_10 = self.transfer_fct(tf.add(tf.matmul(layer_3, weights['h10']), 
                                           biases['b10'])) 
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_10, weights['out_mean']), 
                                 biases['out_mean']))
        return x_reconstr_mean
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})





def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost)
    return vae



#f = gzip.open('mnist.pkl.gz', 'rb')
#train_set, valid_set, test_set = cPickle.load(f)
#f.close()

#data = train_set[0]
#labels = train_set[1]

#n_samples = data.shape[0]

mnist = input_data_cifar.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_recog_3=500, # 3rd layer encoder neurons
         n_hidden_recog_4=500, # 4th layer encoder neurons
         n_hidden_recog_5=500, # 5th layer encoder neurons
         n_hidden_recog_6=500, # 6th layer encoder neurons
         n_hidden_recog_7=500, # 7th layer encoder neurons
         n_hidden_recog_8=500, # 8th layer encoder neurons
         n_hidden_recog_9=500, # 9th layer encoder neurons
         n_hidden_recog_10=500, # 10th layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_hidden_gener_3=500, # 3nd layer decoder neurons
         n_hidden_gener_4=500, # 4th layer decoder neurons
         n_hidden_gener_5=500, # 5th layer decoder neurons
         n_hidden_gener_6=500, # 6th layer decoder neurons
         n_hidden_gener_7=500, # 7th layer decoder neurons
         n_hidden_gener_8=500, # 8th layer decoder neurons
         n_hidden_gener_9=500, # 9th layer decoder neurons
         n_hidden_gener_10=500, # 10th layer decoder neurons
         n_input=(32*32), # MNIST data input (img shape: 28*28)
         n_z=32)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=1000)

saver = tf.train.Saver()
saver.save(vae.sess, 'cif-2l-500n-1000e.ckpt')

