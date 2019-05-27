import tensorflow as tf
import numpy as np
from matting import load_path,load_data,load_alphamatting_data,load_validation_data,unpool
import os
from scipy import misc
import matplotlib.pyplot as plt
from gen_gradient_data import sobel_demo,Scharr_demo,Laplace_demo
os.environ['CUDA_VISIBLE_DEVICES']='0'

image_size = 320
train_batch_size = 1
test_stride_size = 300
max_epochs = 1000000
hard_mode = False

#checkpoint file path
pretrained_model = True
#pretrained_model = False
test_dir = './alhpamatting'
test_outdir = './test_predict'
#validation_dir = '/data/gezheng/data-matting/new2/validation'

#pretrained_vgg_model_path
model_path = './vgg16_weights.npz'
log_dir = 'matting_log'

dataset_alpha = 'train_data/alpha'
dataset_eps = 'train_data/eps'
dataset_BG = 'train_data/bg'

paths_alpha,paths_eps,paths_BG = load_path(dataset_alpha,dataset_eps,dataset_BG,hard_mode = hard_mode)

range_size = len(paths_alpha)
print('range_size is %d' % range_size)
#range_size/batch_size has to be int
batchs_per_epoch = int(range_size/train_batch_size) 

index_queue = tf.train.range_input_producer(range_size, num_epochs=None,shuffle=True, seed=None, capacity=32)
index_dequeue_op = index_queue.dequeue_many(train_batch_size, 'index_dequeue')

image_batch = tf.placeholder(tf.float32, shape=(train_batch_size,image_size,image_size,3))
#raw_RGBs = tf.placeholder(tf.float32, shape=(train_batch_size,image_size,image_size,3))
GT_matte_batch = tf.placeholder(tf.float32, shape = (train_batch_size,image_size,image_size,1))
#GT_trimap = tf.placeholder(tf.float32, shape = (train_batch_size,image_size,image_size,1))
GT_mask = tf.placeholder(tf.float32, shape = (train_batch_size,image_size,image_size,1))
#GTBG_batch = tf.placeholder(tf.float32, shape = (train_batch_size,image_size,image_size,3))
#GTFG_batch = tf.placeholder(tf.float32, shape = (train_batch_size,image_size,image_size,3))
training = tf.placeholder(tf.bool)

tf.add_to_collection('image_batch',image_batch)
#tf.add_to_collection('GT_trimap',GT_trimap)
tf.add_to_collection('GT_mask',GT_mask)
tf.add_to_collection('training',training)

en_parameters = []
pool_parameters = []

b_gradient = tf.identity(image_batch,name = 'b_gradient')
#b_trimap = tf.identity(GT_trimap,name = 'b_trimap')
b_mask = tf.identity(GT_mask,name = 'b_mask')
b_GTmatte = tf.identity(GT_matte_batch,name = 'b_GTmatte')
#b_GTBG = tf.identity(GTBG_batch,name = 'b_GTBG')
#b_GTFG = tf.identity(GTFG_batch,name = 'b_GTFG')

tf.summary.image('GT_matte_batch',b_GTmatte,max_outputs = 5)
#tf.summary.image('trimap',b_trimap,max_outputs = 5)
tf.summary.image('mask',b_mask,max_outputs = 5)
#tf.summary.image('raw_RGBs',raw_RGBs,max_outputs = 5)

#b_input = tf.concat([b_RGB,b_trimap],3)
b_input = tf.concat([b_gradient,b_mask],3)

# conv1_1
with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(b_input, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv1_2
with tf.name_scope('conv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool1
pool1,arg1 = tf.nn.max_pool_with_argmax(conv1_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')
pool_parameters.append(arg1)

# conv2_1
with tf.name_scope('conv2_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv2_2
with tf.name_scope('conv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool2
pool2,arg2 = tf.nn.max_pool_with_argmax(conv2_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool2')
pool_parameters.append(arg2)

# conv3_1
with tf.name_scope('conv3_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv3_2
with tf.name_scope('conv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv3_3
with tf.name_scope('conv3_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool3
pool3,arg3 = tf.nn.max_pool_with_argmax(conv3_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool3')
pool_parameters.append(arg3)

# conv4_1
with tf.name_scope('conv4_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv4_2
with tf.name_scope('conv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv4_3
with tf.name_scope('conv4_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool4
pool4,arg4 = tf.nn.max_pool_with_argmax(conv4_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool4')
pool_parameters.append(arg4)

# conv5_1
with tf.name_scope('conv5_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                         stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                     trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv5_2
with tf.name_scope('conv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv5_3
with tf.name_scope('conv5_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool5
pool5,arg5 = tf.nn.max_pool_with_argmax(conv5_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool5')
pool_parameters.append(arg5)
# conv6_1
with tf.name_scope('conv6_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([7, 7, 512, 4096], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv6_1 = tf.nn.relu(out, name='conv6_1')
    en_parameters += [kernel, biases]
#deconv6
with tf.variable_scope('deconv6') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv6_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv6 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv6')

#deconv5_1/unpooling
deconv5_1 = unpool(deconv6,pool_parameters[-1])

#deconv5_2
with tf.variable_scope('deconv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv5_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv5_2')

#deconv4_1/unpooling
deconv4_1 = unpool(deconv5_2,pool_parameters[-2])

#deconv4_2
with tf.variable_scope('deconv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv4_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv4_2')

#deconv3_1/unpooling
deconv3_1 = unpool(deconv4_2,pool_parameters[-3])

#deconv3_2
with tf.variable_scope('deconv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv3_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv3_2')

#deconv2_1/unpooling
deconv2_1 = unpool(deconv3_2,pool_parameters[-4])

#deconv2_2
with tf.variable_scope('deconv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 128, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv2_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv2_2')

#deconv1_1/unpooling
deconv1_1 = unpool(deconv2_2,pool_parameters[-5])

#deconv1_2
with tf.variable_scope('deconv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv1_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv1_2')
#pred_alpha_matte
with tf.variable_scope('pred_alpha') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 1], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv1_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    pred_mattes = tf.nn.sigmoid(out)

tf.add_to_collection("pred_mattes", pred_mattes)

#wl = tf.where(tf.equal(b_trimap,128),tf.fill([train_batch_size,image_size,image_size,1],1.),tf.fill([train_batch_size,image_size,image_size,1],0.))
wl = tf.where(tf.equal(b_mask,128),tf.fill([train_batch_size,image_size,image_size,1],1.),tf.fill([train_batch_size,image_size,image_size,1],0.))
unknown_region_size = tf.reduce_sum(wl)

tf.summary.image('pred_mattes',pred_mattes,max_outputs = 5)
alpha_diff = tf.sqrt(tf.square(pred_mattes - GT_matte_batch)+ 1e-12)

p_RGB = []
pred_mattes.set_shape([train_batch_size,image_size,image_size,1])
#b_GTBG.set_shape([train_batch_size,image_size,image_size,3])
#b_GTFG.set_shape([train_batch_size,image_size,image_size,3])
#raw_RGBs.set_shape([train_batch_size,image_size,image_size,3])
b_GTmatte.set_shape([train_batch_size,image_size,image_size,1])

#pred_final =  tf.where(tf.equal(b_trimap,128), pred_mattes,b_trimap/255.0)
pred_final =  tf.where(tf.equal(b_mask,128), pred_mattes,b_mask/255.0)
tf.summary.image('pred_final',pred_final,max_outputs = 5)

l_matte = tf.unstack(pred_final)
#BG = tf.unstack(b_GTBG)
#FG = tf.unstack(b_GTFG)

#for i in range(train_batch_size):
#    p_RGB.append(l_matte[i] * FG[i] + (tf.constant(1.) - l_matte[i]) * BG[i])
#pred_RGB = tf.stack(p_RGB)

#tf.summary.image('pred_RGB',pred_RGB,max_outputs = 5)
#c_diff = tf.sqrt(tf.square(pred_RGB - raw_RGBs) + 1e-12)/255.0

alpha_loss = tf.reduce_sum(alpha_diff * wl)/(unknown_region_size)
#comp_loss = tf.reduce_sum(c_diff * wl)/(unknown_region_size)

# tf.summary.image('alpha_diff',alpha_diff * wl_alpha,max_outputs = 5)
# tf.summary.image('c_diff',c_diff * wl_RGB,max_outputs = 5)

tf.summary.scalar('alpha_loss',alpha_loss)
#tf.summary.scalar('comp_loss',comp_loss)

#total_loss = (alpha_loss + comp_loss) * 0.5
total_loss = alpha_loss
tf.summary.scalar('total_loss',total_loss)
global_step = tf.Variable(0,trainable=False)


train_op = tf.train.AdamOptimizer(learning_rate = 1e-5).minimize(total_loss,global_step = global_step)

saver = tf.train.Saver(tf.trainable_variables() , max_to_keep = 1)

coord = tf.train.Coordinator()
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)
#with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(coord=coord,sess=sess)
    batch_num = 0
    epoch_num = 0
    #initialize all parameters in vgg16
    if not pretrained_model:
        weights = np.load(model_path)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i == 28:
                break
            if k == 'conv1_1_W':  
                sess.run(en_parameters[i].assign(np.concatenate([weights[k],np.zeros([3,3,1,64])],axis = 2)))
            else:
                if k=='fc6_W':
                    tmp = np.reshape(weights[k],(7,7,512,4096))
                    sess.run(en_parameters[i].assign(tmp))
                else:
                    sess.run(en_parameters[i].assign(weights[k]))
        print('finish loading vgg16 model')
    else:
        print('Restoring pretrained model...')
        saver.restore(sess,tf.train.latest_checkpoint('./model'))
    sess.graph.finalize()

    nummm=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    fff=-1
    while epoch_num < max_epochs:  
        while batch_num < batchs_per_epoch:
            fff=fff+1
            batch_index = sess.run(index_dequeue_op)

            batch_alpha_paths = paths_alpha[batch_index]
            batch_eps_paths = paths_eps[batch_index]
            batch_BG_paths = paths_BG[batch_index]
#            batch_RGBs,batch_trimaps,batch_alphas,batch_BGs,batch_FGs,RGBs_with_mean = load_data(batch_alpha_paths,batch_eps_paths,batch_BG_paths)
            batch_Gradients,batch_masks,batch_alphas = load_data(batch_alpha_paths,batch_eps_paths)

#            feed = {image_batch:batch_RGBs, GT_matte_batch:batch_alphas,GT_trimap:batch_trimaps, GTBG_batch:batch_BGs, GTFG_batch:batch_FGs,raw_RGBs:RGBs_with_mean,training:True}
            feed = {image_batch:batch_Gradients, GT_matte_batch:batch_alphas,GT_mask:batch_masks,training:True}

            _,loss,summary_str,step= sess.run([train_op,total_loss,summary_op,global_step],feed_dict = feed)
#            trimap=b_trimap.eval(feed_dict=feed)
            mask_=b_mask.eval(feed_dict=feed)
            alpha_=pred_final.eval(feed_dict=feed)
            print('epoch %d   batch %d   loss is %f' %(epoch_num,batch_num,loss))
            
#            plt.figure(figsize=(20, 100))
#            plt.subplot(1,4,1)
 #           plt.imshow(batch_Gradients.reshape(320,320,3))
            misc.imsave(os.path.join('./tmp/',str(nummm[fff%20])+'_1grad.png'),batch_Gradients.reshape(320,320,3))
 #           plt.subplot(1,4,2)
  #          plt.imshow(np.concatenate([mask_.reshape(320,320,1)/255,mask_.reshape(320,320,1)/255,mask_.reshape(320,320,1)/255],axis=2))
            misc.imsave(os.path.join('./tmp/',str(nummm[fff%20])+'_2mask.png'),np.concatenate([mask_.reshape(320,320,1)/255,mask_.reshape(320,320,1)/255,mask_.reshape(320,320,1)/255],axis=2))
  #          plt.subplot(1,4,3)
  #          plt.imshow(np.concatenate([alpha_.reshape(320,320,1),alpha_.reshape(320,320,1),alpha_.reshape(320,320,1)],axis=2))
            misc.imsave(os.path.join('./tmp/',str(nummm[fff%20])+'_3res.png'),np.concatenate([alpha_.reshape(320,320,1),alpha_.reshape(320,320,1),alpha_.reshape(320,320,1)],axis=2))
  #          plt.subplot(1,4,4)
  #          plt.imshow(np.concatenate([batch_alphas.reshape(320,320,1),batch_alphas.reshape(320,320,1),batch_alphas.reshape(320,320,1)],axis=2))
            misc.imsave(os.path.join('./tmp/',str(nummm[fff%20])+'_4alpha.png'),np.concatenate([batch_alphas.reshape(320,320,1),batch_alphas.reshape(320,320,1),batch_alphas.reshape(320,320,1)],axis=2))
 #           plt.show()

            
            if step%200 == 0:
#            if 1:
                print('saving model......')
                saver.save(sess,'./model/model.ckpt',global_step = step, write_meta_graph = False)

                print('test on validation data...')
                vali_diff = []
#                test_RGBs,test_trimaps,test_alphas,all_shape,image_paths,trimap_size= load_alphamatting_data(test_dir)
#                test_Gradients,test_trimaps,test_alphas,all_shape,image_paths,trimap_size= load_alphamatting_data(test_dir)
                
                rgb_path = os.path.join(test_dir,'rgb')
                trimap_path = os.path.join(test_dir,'trimap')
                alpha_path = os.path.join(test_dir,'alpha')	
                images = os.listdir(trimap_path)
                test_num = len(images)
            
                for i in range(test_num):
                    rgb = misc.imread(os.path.join(rgb_path,images[i]))
                    trimap = misc.imread(os.path.join(trimap_path,images[i]),'L')
                    alpha = misc.imread(os.path.join(alpha_path,images[i]),'L')/255.0
                    shape_i = trimap.shape
                    mask_grad = alpha==0
                    img_gradient = sobel_demo(rgb,mask_grad)/255
                    test_Gradient = img_gradient.astype(np.float32)
                    trimap = trimap.astype(np.float32)
                    test_trimap = np.expand_dims(trimap,2)
                    test_alpha = np.expand_dims(alpha,2)
                    image_path = images[i]
                    trimap_size = np.sum(trimap==128)
                                    
                    res_=np.zeros(((shape_i[0],shape_i[1],1)))
                    sum_=np.zeros(((shape_i[0],shape_i[1],1)))
                    
                    hw_index=[]
                    test_grad_patch=[]
                    test_trimap_patch=[]
                    test_label_patch=[]
                    h=0
                    while h <= shape_i[0]-image_size:
        #                print("h: ", h)
                        w=0
                        while w <= shape_i[1]-image_size:
                            test_grad_patch_1=test_Gradient[h:h+image_size, w:w+image_size, :]
                            test_trimap_patch_1=test_trimap[h:h+image_size, w:w+image_size, :]
                            test_label_patch_1=test_alpha[h:h+image_size, w:w+image_size, :]
                            
                            hw_index.append([h,w])
                            test_grad_patch.append(test_grad_patch_1)
                            test_trimap_patch.append(test_trimap_patch_1)
                            test_label_patch.append(test_label_patch_1)
                            
                            if(len(hw_index)==train_batch_size):
                                test_grad_patch=np.array(test_grad_patch)
                                test_trimap_patch=np.array(test_trimap_patch)
                                test_label_patch=np.array(test_label_patch)
                                
                                feed = {image_batch:test_grad_patch, GT_matte_batch:test_label_patch,GT_mask:test_trimap_patch,training:False}
                                test_out = sess.run(pred_final,feed_dict = feed)
                                
                                for res_index in range(train_batch_size):
                                    h_cur=hw_index[res_index][0]
                                    w_cur=hw_index[res_index][1]
                                    res_[h_cur:h_cur+image_size, w_cur:w_cur+image_size, :]+=test_out[res_index,:,:,:].reshape(image_size,image_size,1)
                                    plt.figure(figsize=(20, 100))
                                    plt.subplot(1,4,1)
                                    plt.imshow(test_grad_patch.reshape(320,320,3))
                                    plt.subplot(1,4,2)
                                    plt.imshow(np.concatenate([test_trimap_patch.reshape(320,320,1)/255,test_trimap_patch.reshape(320,320,1)/255,test_trimap_patch.reshape(320,320,1)/255],axis=2))
                                    plt.subplot(1,4,3)
                                    plt.imshow(np.concatenate([test_out[res_index,:,:,:].reshape(image_size,image_size,1),test_out[res_index,:,:,:].reshape(image_size,image_size,1),test_out[res_index,:,:,:].reshape(image_size,image_size,1)],axis=2))
                                    plt.subplot(1,4,4)
                                    plt.imshow(np.concatenate([test_label_patch.reshape(320,320,1),test_label_patch.reshape(320,320,1),test_label_patch.reshape(320,320,1)],axis=2))
                                    plt.show()
                                    sum_[h_cur:h_cur+image_size, w_cur:w_cur+image_size, :]+=1
                                hw_index=[]
                                test_grad_patch=[]
                                test_trimap_patch=[]
                                test_label_patch=[]
                            
                            if w+test_stride_size>shape_i[1]-image_size and w+image_size!=shape_i[1]:
                                w=shape_i[1]-image_size-test_stride_size

                            w+=test_stride_size
                        
                        if h+test_stride_size>shape_i[0]-image_size and h+image_size!=shape_i[0]:
                            h=shape_i[0]-image_size-test_stride_size
                            
        #                if h%100==0:
        #                    print(h)
                        h+=test_stride_size
                       
                    if len(hw_index):
                        cur_num=len(hw_index)
                        for t in range(cur_num,train_batch_size):
                            hw_index.append(hw_index[t%cur_num])
                            test_grad_patch.append(test_grad_patch[t%cur_num])
                            test_trimap_patch.append(test_trimap_patch[t%cur_num])
                            test_label_patch.append(test_label_patch[t%cur_num])
                        test_grad_patch=np.array(test_grad_patch)
                        test_trimap_patch=np.array(test_trimap_patch)
                        test_label_patch=np.array(test_label_patch)
                        
                        feed = {image_batch:test_grad_patch, GT_matte_batch:test_label_patch,GT_mask:test_trimap_patch,training:False}
                        test_out = sess.run(pred_final,feed_dict = feed)
                        
                        for res_index in range(cur_num):
                            h_cur=hw_index[res_index][0]
                            w_cur=hw_index[res_index][1]
                            res_[h_cur:h_cur+image_size, w_cur:w_cur+image_size, :]+=test_out[res_index,:,:,:].reshape(image_size,image_size,1)
                            sum_[h_cur:h_cur+image_size, w_cur:w_cur+image_size, :]+=1
                    hw_index=[]
                    test_grad_patch=[]
                    test_trimap_patch=[]
                    test_label_patch=[]
                    
                    res_ = res_/sum_
                                        
                    i_out = res_
                    vali_diff.append(np.sum(np.abs(i_out-test_alpha))/trimap_size)
                    misc.imsave(os.path.join(test_outdir,image_path),i_out.reshape(shape_i))
                
                vali_loss = np.mean(vali_diff)
                print('validation loss is '+ str(vali_loss))
                validation_summary = tf.Summary()
                validation_summary.value.add(tag='validation_loss',simple_value = vali_loss)
                summary_writer.add_summary(validation_summary,step)

            summary_writer.add_summary(summary_str,global_step = step)
            batch_num += 1
        batch_num = 0
        epoch_num += 1


