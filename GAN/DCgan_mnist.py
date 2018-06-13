import numpy as np 
import tensorflow as tf 
import pickle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/MNIST_data/')

#get input data
def get_inputs(noise_dim,image_height,image_width,image_depth):
    inputs_real=tf.placeholder(tf.float32,[None,image_height,image_width,image_depth],name='inputs_real')
    inputs_noise=tf.placeholder(tf.float32,[None,noise_dim],name='inputs_noise')
    return inputs_real,inputs_noise

#get generator
def get_generator(noise_img,output_dim,is_train=True,alpha=0.01):
    with tf.variable_scope('generator',reuse=(not is_train)):
        #from 100*1 to 4*4*512
        layer1=tf.layers.dense(noise_img,4*4*512)
        layer1=tf.reshape(layer1,[-1,4,4,512])
        layer1=tf.layers.batch_normalization(layer1,training=is_train)
        layer1=tf.maximum(alpha*layer1,layer1)
        layer1=tf.nn.dropout(layer1,keep_prob=0.8)
        #from 4*4*512 to 7*7*256
        layer2=tf.layers.conv2d_transpose(layer1,256,4,strides=1,padding='valid')
        layer2=tf.layers.batch_normalization(layer2,training=is_train)
        layer2=tf.maximum(alpha*layer2,layer2)
        layer2=tf.nn.dropout(layer2,keep_prob=0.8)
        #from 7*7*256 to 14*14*128
        layer3=tf.layers.conv2d_transpose(layer2,128,3,strides=2,padding'same')
        layer3=tf.layers.batch_normalization(layer3,training=is_train)
        layer3=tf.maximum(alpha*layer3,layer3)
        layer3=tf.nn.dropout(layer3,keep_prob=0.8)
        #from 14*14*128 to 28*28*1
       logits= tf.layers.conv2d_transpose(layer3,output_dim,3,strides=2,padding='same')
       #mnist dataset from [0,1]   to [-1,1]
       outputs=tf.tanh(logits)
       return outputs

#get discriminator
def get_discriminator(inputs_img,reuse=False,alpha=0.01):
    with tf.variable_scope('discriminator',reuse=reuse):
        #from 28*28*1 to 14*14*128
        layer1=tf.layers.conv2d(inputs_img,128,3,strides=2,padding='same')
        layer1=tf.maximum(alpha*layer1,layer1)
        layer1=tf.nn.dropout(layer1,keep_prob=0.8)
        #from 14*14*128 to 7*7*256
        layer2=tf.layers.conv2d(layer1,256,3,strides=2,padding='same')
        layer2=tf.layers.batch_normalization(layer2,training=True)
        layer2=tf.maximum(alpha*layer2,layer2)
        layer2=tf.nn.dropout(layer2,keep_out=0.8)
        #from 7*7*256 to 4*4*512
        layer3=tf.layers.conv2d(layer2,512,3,strides=2,padding='same')
        layer3=tf.layers.batch_normalization(layer3,training=True)
        layer3=tf.maximum(alpha*layer3,layer3)
        layer3=tf.nn.dropout(layer3,keep_out=0.8)
        #from 4*4*512 to (4*4*512)*1
        flatten=tf.reshape(layer3,(-1,4*4*512))
        logits=tf.layers.dense(flatten,1)
        outputs=tf.sigmoid(logits)
        return logits,outputs

#get loss
def get_loss(inputs_real,inputs_noise,image_depth,smooth=0.1):
    g_outputs=get_generator(inputs_noise,image_depth,is_train=True)
    d_logits_real,d_outputs_real=get_discriminator(inputs_real)
    d_logits_fake,d_outputs_fake=get_discriminator(g_outputs,reuse=True)
    g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_outputs_fake)*(1-smooth)))
    d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_outputs_real)*(1-smooth)))
    d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_outputs_fake)))
    d_loss=tf.add(d_loss_real,d_loss_fake)
    return g_loss,d_loss

#get optimizer
def get_optimizer(g_loss,d_loss,betal=0.4,learning_rate=0.001):
    train_vars=tf.trainable_variables()
    g_vars=[var for var in train_vars if var.name.startswith('generator')]
    d_vars=[var for var in train_vars if var.name.startswith('discriminator')]
    #optimizer
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
       g_opt=tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list=g_vars) 
       d_opt=tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=d_vars)
    return g_opt,d_opt 

#plot images
def plot_images(samples):
    fig,axes=plt.subplots(nrows=1,ncols=25,sharex=True,sharey=True,figsize=(50,2))
    for img,ax in zip(samples,axes):
        ax.imshow(img.reshape((28,28)),cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)

#show generator output
def show_generator_output(sess,n_images,inputs_noise,output_dim):
    cmap='Greys_r'
    noise_shape=inputs_noise.get_shape().as_list()[-1]
    #generate noise picture
    examples_noise=np.random.uniform(-1,1,size=[n_images,noise_shape])
    samples=sess.run(get_generator(inputs_noise,output_dim,False),feed_dict={inputs_noise:examples_noise})
    result=np.squeeze(samples,-1)
    return result

#train network
def train(noise_size,data_shape,batch_size,n_samples):
    #save loss
    losses=[]
    steps=0
    inputs_real,inputs_noise=get_inputs(noise,data_shape[1],data_shape[2],data_shape[3])
    g_loss,d_loss=get_loss(inputs_real,inputs_noise,data_shape[-1])
    g_train_opt,d_train_opt=get_optimizer(g_loss,d_loss,betal,learning_rate)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #epoch
        for e in range(epochs):
            for batch_i in range(mnist.train.num_examples//batch_size):
                steps+=1
                batch=mnist.train.next_batch(batch_size)
                batch_images=batch[0].reshape((batch_size,data_shape[1],data_shape[2],data_shape[3]))
                #scale from [0,1] to [-1,1]
                batch_images=batch_images*2-1
                #noise
                batch_noise=np.random.uniform(-1,1,size(batch_size,noise_size))
                #run optimizer
                _=sess.run(g_train_opt,feed_dict={inputs_real:batch_images,inputs_noise:batch_noise})
                _=sess.run(d_train_opt,feed_dict={inputs_real:batch_images,inputs_noise:batch_noise})
                if steps%101==0:
                    train_loss_d=d_loss.eval({inputs_real:batch_images,inputs_noise:batch_noise})
                    train_loss_g=g_loss.eval({inputs_real:batch_images,inputs_noise:batch_noise})
                    losses.append((train_loss_d,train_loss_g))
                    #show picture
                    samples=show_generator_output(sess,n_samples,inputs_noise,data_shape[-1])
                    plot_images(samples)
                    print("Epoch {}/{}...".format(e+1,epochs),"Discriminator Loss: {:.4f}...".format(train_loss_d),"Generator Loss: {:.4f}...".format(train_loss_g))

if __name__=='__main__':
    #define params
    batch_size=64
    noise_size=100
    epochs=5
    n_samples=25
    learning_rate=0.001
    with tf.Graph().as_default():
        train(noise_size,[-1,28,28,1],batch_size,n_samples)
