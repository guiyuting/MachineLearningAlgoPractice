'''
vanilla vae in mnist
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
from PIL import Image
import os
import numpy as np

def dataprepare():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    return mnist.train, mnist.validation, mnist.test

def displayimg(imgs, batch, name):
    row = int(len(imgs) / 10)
    newarray = np.zeros((1,1))
    for r in range(row):
        row_img = np.concatenate(imgs[r*10:(r+1)*10],0)
        if r == 0:
            newarray = row_img
        else:
            newarray = np.concatenate((newarray, row_img), 1)
    newimg = Image.fromarray(newarray)
    newimg.save(name + "_" + str(batch) + ".tiff")

class vae():
    def __init__(self):
        
        self.input_size    = 784 # image size 28*28 
        self.hidden_dim    = 400 # size for hidden layer
        self.latent_dim    = 20  # latent mu and sigma size, mixture of gaussian
        self.learning_rate = 0.01
        self.epochs        = 10
        self.batch_size    = 100

        # initiate trained variable
        self.x = tf.placeholder(tf.float32, [None, self.input_size])

        # encoder weight and bias
        # input-hidden input_size-hidden_dim
        self.W_ih = tf.Variable(tf.random_normal([self.input_size, self.hidden_dim],\
                   stddev = 0.001))
        
        self.b_ih = tf.Variable(tf.constant(0., shape = [self.hidden_dim]))

        #hidden-latentmu and latend sigma  hidden_dim-latent_dim
        self.W_hmean = tf.Variable(tf.random_normal([self.hidden_dim, self.latent_dim],\
                    stddev = 0.001)) 
        self.b_hmean = tf.Variable(tf.constant(0., shape = [self.latent_dim]))
        
        # variance in log scale to make sure it has similar scale with mean 
        self.W_hlogvar = tf.Variable(tf.random_normal([self.hidden_dim, self.latent_dim],\
                    stddev = 0.001)) 
        self.b_hlogvar = tf.Variable(tf.constant(0., shape = [self.latent_dim]))
        


        # decoder weight and bias
        # latent-hidden
        self.W_lh = tf.Variable(tf.random_normal([self.latent_dim, self.hidden_dim],\
                   stddev = 0.001))
        self.b_lh = tf.Variable(tf.constant(0., shape= [self.hidden_dim]))
        
        
        # hidden-input_size
        self.W_hi = tf.Variable(tf.random_normal([self.hidden_dim, self.input_size],\
                    stddev = 0.001))
        self.b_hi = tf.Variable(tf.constant(0., shape = [self.input_size]))

        
        # autoencoder, add relu-converge fast
        hidden_enc = tf.sigmoid(tf.matmul(self.x, self.W_ih) + self.b_ih) # TODO
        latent_mean = tf.matmul(hidden_enc, self.W_hmean) + self.b_hmean
        latent_logvar  = tf.matmul(hidden_enc, self.W_hlogvar) + self.b_hlogvar
        epsilon = tf.Variable(tf.random_normal([self.latent_dim]))
        z = latent_mean + tf.sqrt(tf.exp(latent_logvar)) * epsilon

        hidden_dec = tf.sigmoid(tf.matmul(z, self.W_lh) + self.b_lh)  # TODO
        self.recon_x = tf.matmul(hidden_dec, self.W_hi) + self.b_hi # 100 * 784
        
        # KL div and reconstruct loss
        self.KL = -0.5 * tf.reduce_sum(1.0 +  latent_logvar - tf.pow(latent_mean, 2) - tf.exp(latent_logvar), reduction_indices = 1) 
        # batchsize * 1
        
        self.logit = tf.nn.sigmoid_cross_entropy_with_logits(self.recon_x, self.x)
        # inputsize * 1 

        self.reconstruct_loss = tf.reduce_sum(self.logit) 
        # wrt pixels

        self.loss = tf.reduce_mean( self.KL + 0.5 * self.reconstruct_loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=0.8).minimize(self.loss)

        self.init = tf.initialize_all_variables()

        self.sess = None


    def train(self, training_set):
        self.sess = tf.Session()
        self.sess.run(self.init)
        print("Training")
        for step in xrange(100000):
            batch = training_set.next_batch(self.batch_size)
            feed = {self.x: batch[0]}
            _, c, KL, recon, logit = self.sess.run([self.optimizer, self.loss, self.KL, self.reconstruct_loss, self.logit], feed_dict = feed)
            if step % 50 == 0:
                print("Step: %s, Cost: %s" %(step, recon))
            if step % 1000 == 0: 
                recon = self.sess.run([self.recon_x], feed_dict = feed)
                displayimg(recon[0].reshape(self.batch_size, 28, 28), step, "recon")
                displayimg(batch[0].reshape(self.batch_size, 28, 28), step, "x")

        

def main():
    training, valid, test = dataprepare()
    model = vae()
    model.train(training)

if __name__ == "__main__":
    main()
            




    


