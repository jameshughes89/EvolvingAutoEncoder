#============================================================================
# Name        : lilGA.py
# Author      : James Alexander Hughes
# Version     : 0.1
# Copyright   : Fek Erf
# Description : Hello World in C++, Ansi-style
#============================================================================

import numpy as np
import scipy
import scipy.spatial
import scipy.stats
import sys
import threading
import matplotlib.pylab as plt
import gzip
import cPickle
import tensorflow as tf
import input_data

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
                 learning_rate=0.001, batch_size=100):
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
            
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
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
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
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
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
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




################################################

def distFromPoint(c):
	enc = vae.transform([c])
	return scipy.spatial.distance.euclidean(point, enc)


def distFromPointPlusEntropy(c):
	enc = vae.transform([c])
	return scipy.spatial.distance.euclidean(point, enc) + scipy.stats.entropy(c)


def countNotZero(c):
	numNotZero = len(c)	
	for g in c:
		if g == 0:
			numNotZero -= 1

	return numNotZero

def closeToZero(c):
	return np.sum(c)


def calculatePopulationFitness(pop, popFits, fitFunction):
	for i in range(len(pop)):
		popFits[i] = fitFunction(pop[i])


def findBestChromosomeIndex(popFits):
	return np.argmin(popFits)			#useless


def selection(popFits):
	first = np.random.randint(len(popFits))
	second = np.random.randint(len(popFits))
	
	if popFits[first] < popFits[second]:
		return first
	else:
		return second

def selectionWithEntropyFirst(popFits, pop, EPSILON = 0.025):
	first = np.random.randint(len(popFits))
	second = np.random.randint(len(popFits))

	if np.abs(scipy.stats.entropy(pop[first]) - scipy.stats.entropy(pop[second])) < EPSILON:
		if popFits[first] < popFits[second]:
			return first
		else:
			return second
	else:
		if scipy.stats.entropy(pop[first]) < scipy.stats.entropy(pop[second]):
			return first
		else:
			return second

def selectionWithEntropySecond(popFits, pop, EPSILON = 0.2):
	first = np.random.randint(len(popFits))
	second = np.random.randint(len(popFits))

	if np.abs(popFits[first] - popFits[second]) < EPSILON:
		if scipy.stats.entropy(pop[first], pop[second]) < scipy.stats.entropy(pop[second], pop[first]):
			return first
		else:
			return second
	else:
		if popFits[first] < popFits[second]:
			return first
		else:
			return second

def selectionWithEntropyBeingMainFit(popFits, pop, EPSILON = 0.02):
	first = np.random.randint(len(popFits))
	second = np.random.randint(len(popFits))

	if np.abs(popFits[first] - popFits[second]) < EPSILON:
		if distFromPoint(pop[first]) < distFromPoint(pop[second]):
			return first
		else:
			return second
	else:
		if popFits[first] < popFits[second]:
			return first
		else:
			return second


def opXover(c1, c2):
	point = np.random.randint(len(c1))
	#tmp = np.copy(c1[point:])
	#c1[point:] = np.copy(c2[point:])
	#c2[point:] = np.copy(tmp)
	for i in range(point, len(c1)):
		c1[i], c2[i] = c2[i], c1[i]

def randomMutation(c):
	#c[np.random.randint(len(c))] = np.random.randint(256)
	c[np.random.randint(len(c))] = np.random.rand()

	
def makePopulation(popSize, numIslands, chromoLength):
	curPop = []
	nextPop = []
	for i in range(numIslands):
		curIsland = []
		nextIsland = []
		for k in range(popSize):
			#curIsland.append(np.random.randint(0,256,chromoLength).astype('u1'))
			#nextIsland.append(np.random.randint(0,256,chromoLength).astype('u1'))
			curIsland.append(np.random.rand(chromoLength))
			nextIsland.append(np.random.rand(chromoLength))
	
		curPop.append(np.array(curIsland))
		nextPop.append(np.array(nextIsland))
	return np.array(curPop), np.array(nextPop), np.zeros((numIslands, popSize))


def mating(curPop, nextPop, curPopFits):
	#calculatePopulationFitness(curPop, curPopFits, distFromPoint)
	#calculatePopulationFitness(curPop, curPopFits, distFromPointPlusEntropy)
	calculatePopulationFitness(curPop, curPopFits, scipy.stats.entropy)
	
	nextPop[0] = np.copy(curPop[findBestChromosomeIndex(curPopFits)])
	
	for i in range(1,len(curPop), 2):
		#first = selection(curPopFits)
		#second = selection(curPopFits)
		first = selectionWithEntropyBeingMainFit(curPopFits, curPop, EPSILON=0.05)
		second = selectionWithEntropyBeingMainFit(curPopFits, curPop, EPSILON=0.05)

		nextPop[i] = np.copy(curPop[first])
		nextPop[i+1] = np.copy(curPop[second])

		if np.random.rand() < CROSSOVER_RATE:
			opXover(nextPop[i], nextPop[i+1])

		for j in range(i, i + 2):
			for k in range(MUTATION_CHANCES):
				if np.random.rand() < MUTATION_RATE:
					randomMutation(nextPop[i])
				

		for i in range(len(curPop)):
			curPop[i], nextPop[i] = nextPop[i], curPop[i]

def evolve(curPop, nextPop, curPopFits):
	for i in range(NUMBER_GENERATIONS/POP_SIZE):
		mating(curPop, nextPop, curPopFits)


def printPicture(c):
	import matplotlib.pylab as plt
	#img = []
	#for i in range(len(c)):
	#	img.append([c[i], c[i], c[i]])

	#img = np.array(img)
	img = c.reshape((int(CHROMOSOME_SIZE**(0.5)),int(CHROMOSOME_SIZE**(0.5)))) 	
	plt.imshow(img)
	plt.show()

#########################################

#point = np.array([ 1.30678296,  0.00420067, -1.16576886,  0.01931952, -0.04719706, -0.24724936, -0.93993825,  0.01407583,  0.01180963,  0.83896232, 0.00774858,  0.04520635, -0.028522, -0.55640572,  1.48450494, -1.09499741, -0.79522842, -0.38319138, -0.00945271,  0.52145159])
#point = np.array([ 0.59183943, -0.00227089, -0.7524488 ,  0.01478681, -0.02374384,
#    -0.37750342, -0.31698442,  0.01858169,  0.01317838,  0.85508072,
#     0.01039285,  0.03133727, -0.03677523,  0.1895496 ,  0.72862774,
#    -0.9119103 , -0.78050435,  0.64618486, -0.02531429,  0.28865847])	
#point = np.array([ 1.13159215,  0.00266721, -1.20985496,  0.00801371, -0.05183448,
#     0.03617568, -0.32890797,  0.00559942,  0.0012916 ,  0.61611915,
#     0.01695263,  0.02495181, -0.02529396, -0.68418634,  1.10197949,
#    -0.57933033,  0.02450088, -0.9781692 , -0.0202379 ,  0.69870818])	
point = np.array([ -1.23104043e-01,  -8.74245074e-03,  -3.39128792e-01, 1.02540916e-02,  -2.90621072e-04,  -5.07757485e-01, 3.05969387e-01,   2.30875406e-02,   1.45471357e-02, 8.71199191e-01,   1.30371097e-02,   1.74681935e-02, -4.50284593e-02,   9.35504913e-01,  -2.72494256e-02, -7.28823185e-01,  -7.65780210e-01,   1.67556107e+00, -4.11758609e-02,   5.58653474e-02])


CROSSOVER_RATE = .8
MUTATION_RATE = .1
MUTATION_CHANCES =15

CHROMOSOME_SIZE = 28*28
POP_SIZE = 25
NUMBER_ISLANDS = 8
NUMBER_MIGRATIONS = 100
NUMBER_GENERATIONS = 2500

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=0)

saver = tf.train.Saver()
saver.restore(vae.sess, '500.ckpt')


currentPopulations, nextPopulations, populationFitnesss = makePopulation(POP_SIZE, NUMBER_ISLANDS, CHROMOSOME_SIZE)

#printPicture(currentPopulations[0][0])

for j in range(NUMBER_MIGRATIONS):
	THREADS = []
	for i in range(NUMBER_ISLANDS):
		THREADS.append(threading.Thread(target=evolve, args=(currentPopulations[i], nextPopulations[i], populationFitnesss[i])))
		THREADS[i].start()

	for t in THREADS:
		t.join()
	
	s = str(j) + ':\t'
	for i in range(NUMBER_ISLANDS):
		#s = s + str(populationFitnesss[i][0]) + ' (' + str(scipy.stats.entropy(currentPopulations[i][0])) + ')' + '\t'
		s = s + str(populationFitnesss[i][0]) + ' (' + str(distFromPoint(currentPopulations[i][0])) + ')' + '\t'
	print s

	currentPopulations = currentPopulations.reshape((NUMBER_ISLANDS * POP_SIZE, CHROMOSOME_SIZE), order='F')
	currentPopulations = currentPopulations[np.random.permutation(len(currentPopulations)), :]
	currentPopulations = currentPopulations.reshape((NUMBER_ISLANDS, POP_SIZE, CHROMOSOME_SIZE), order='F')
	

#printPicture(currentPopulations[0][0])
oFile = open('lessGen500.csv','w')

for i in range(len(currentPopulations)):
	for j in range(len(currentPopulations[i])):
		for k in range(len(currentPopulations[i][j])):
			oFile.write(str(currentPopulations[i][j][k]) + ', ')
		oFile.write('\n')
	oFile.write('\n\n')

##what we evolved
plt.subplot(3,1,1)
plt.imshow(currentPopulations[0][0].reshape((28,28)))
#plt.show()

#what we evolved after going through network
plt.subplot(3,1,2)
img = vae.reconstruct([currentPopulations[0][0]])
plt.imshow(img[0].reshape((28,28)))
#plt.show()

#what our target is
plt.subplot(3,1,3)
target = vae.generate([point])
plt.imshow(target.reshape((28,28)))
plt.show()

