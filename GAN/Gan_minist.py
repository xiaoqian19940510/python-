import tensorflow as tf
import numpy as np 
import pickle
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/data')

#真实数据和噪音数据
def get_inputs(real_size,noise_size):
    real_img=tf.placeholder(tf.float32,[None,real_size])
    noise_img=tf.placeholder(tf.float32,[None,noise_size])
    return real_img,noise_img

#generator
def get_generator(noise_img,n_units,out_dim,reuse=False,alpha=0.01):
    with tf.variable_scope("generator",reuse=reuse):
        #hidden layer
        hidden1=tf.layers.dense(noise_img,n_units)
        #leaky ReLU
        hidden1=tf.maximum(alpha*hidden1,hidden1)
        #dropout
        hidden1=tf.layers.dropout(hidden1,rate=0.2)
        #logits&outputs
        logits=tf.layers.dense(hidden1,out_dim)
        outputs=tf.tanh(logits)
        return logits,outputs

#discriminator
def get_discriminator(img,n_units,reuse=False,alpha=0.01):
    with tf.variable_scope('discriminator',reuse=reuse):
        #hidden layer
        hidden1=tf.layers.dense(img,n_units)
        #leaky ReLU
        hidden1=tf.maxium(alpha*hidden1,hidden1)
        #logits &outputs
        logits=tf.layers.dense(hidden1,1)
        outputs=tf.sigmoid(logits)
        return logits,outputs


#params define
img_size=mnist.train.images[0].shape[0]
noise_size=100
g_units=128
d_units=128
learning_rate=0.001
alpha=0.01

#construct gan network
tf.reset_default_graph()
real_img,noise_img=get_inputs(real_size,noise_size)
#generator
g_logits,g_outputs=get_generator(noise_img,g_units,img_size)
#discriminator
d_logits_real,d_outputs_real=get_discriminator(real_img,d_units)
d_logits_fake,d_outputs_fake=get_discriminator(g_outputs,d_units,reuse=True)

#discriminator'loss
#recognise real images
d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_logits_real)))
#recognise fake images
d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_logits_fake)))
#full lose
d_loss=tf.add(d_loss_real,d_loss_fake)
#generator loss
g_loss=tf.reduce_mean(tf.nnsigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_logits_fake)))

#...optimizer module...
train_vars=tf.trainable_variables()
#generator
g_vars=[var for var in train_vars if var.name.startswith("generator")]
#discriminator
d_vars=[var for var in train_vars if var.name.startswith("discriminator")]
#optimizer
d_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=d_vars)
g_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list=g_vars)

#...train module...
batch_size=64
#train epochs
epochs=300
#select samples' number
n_sample=25
#save test samples
samples=[]
#save loss
losses=[]
#save generator variables
saver=tf.train.Saver(var_list=g_vars)
#train beginning
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for batch_i in range(mnist.train.num_examples//batch_size):
            batch=mnist.train.next_batch(batch_size)
            batch_images=batch[0].reshape((batch_size,784))
            #tanh output between -1 and 1 ,real and fake share discriminator's params
            batch_images=batch_images*2-1
            #generator's input noise
            batch_noise=np.random.uniform(-1,1,size=(batch_size,noise_size))
            #run optimizers
            _=sess.run(d_train_opt,feed_dict={real_img:batch_images,noise_img:batch_noise})
            _=sess.run(g_train_opt,feed_dict={noise_img:batch_noise})
        #caculate loss of each epoch
        train_loss_d=sess.run(d_loss,feed_dict={real_img:batch_images,noise_img:batch_noise})
        #real loss
        train_loss_d_real=sess.run(d_loss_real,feed_dict={real_img:batch_images,nosie_img:batch_noise})
        #fake loss
        train_loss_d_fake=sess.run(d_loss_fake,feed_dict={real_img:batch_images,noise_img:batch_noise})
        #generator loss
        train_loss_g=sess.run(g_loss,feed_dict={noise_img:batch_noise})
        #print loss
        print("Epoch {}/{}...".format(e+1,epochs),
        "discriminator's loss:{:.4f}(real loss {:.4f} + fake loss {:.4f})".format(train_loss_d,train_loss_d_real,train_loss_d_fake),
        "generator's loss:{:.4f}".format(train_loss_g))
        losses.append(train_loss_d,train_loss_d_real,train_loss_d_fake,train_loss_g)
        #save samples
        sample_noise=np.random.uniform(-1,1,size={n_sample,noise_size})
        gen_samples=sess.run(get_generator(noise_img,g_units,img_size,reuse=True),
        feed_dict={noise_img:sample_noise})
        samples.append(gen_samples)
        saver.save(sess,'./checkpoints/generator.ckpt')
#save to desktop
with open('train_samples.pkl','wb') as f:
    pickle.dump(samples,f)

#plot figure
fig,ax=plt.subplots(figsize=(20,7))
losses=np.array(losses)
plt.plot(losses.T[0],label='discriminator all loss')
plt.plot(losses.T[1],label='discriminator real loss')
plt.plot(losses.T[2],lable='discriminator fake loss')
plt.plot(losses.T[3],label='generator loss')
plt.title('gan network')
ax.set_xlabel('epoch')
plt.legend()



# load samples from generator taken while training
with open('train_samples.pkl','rb') as f:
    samples=pickle.load(f)

def view_samples(epoch,samples):
    fig,axes=plt.subplots(figsize=(7,7),nrows=5,ncols=5,sharey=True,sharex=True)
    #samples[epoch][1] is generator img result
    for ax,img in zip(axes.flatten(),samples[epoch][1]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im=ax.imshow(img.reshape((28,28)),cmap='Greys_r')
    return fig,axes
#show the generator result
_=view_samples(-1,samples)

#...show the all generator process...
#check some epochs
epoch_idx=[10,30,60,90,120,150,180,210,240,290]
show_imgs=[]
for i in epoch_idx:
    show_imgs.append(samples[i][1])

#define the image shape
rows,cols=10,25
fig,axes=plt.subplots(figsize=(30,12),nrows=rows,ncols=cols,sharex=True,sharey=True)
idx=range(0,epochs,int(epochs/rows))
for sample,ax_row in zip(show_imgs,axes):
    for img,ax in zip(sample[::int(len(sample)/cols),ax_row]):
        ax.imshow(img.reshape((28,28)),cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

#generate new pictures
saver=tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('checkpoints'))
    sample_noise=np.random.uniform(-1,1,size={25,noise_size})
    gen_samples=sess.run(get_generator(noise_img,g_units,img_size,reuse=True),
    feed_dict={noise_img:sample_noise})

_=view_samples(0,[gen_samples])
