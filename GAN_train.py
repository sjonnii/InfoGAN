import tensorflow as tf
from GAN_Model import Model
import input_data
import os
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import create_plots as plots
import time


FLAGS = tf.app.flags.FLAGS

def train():
    data = input_data.read_data_sets('unlabelled_cells_filtered_newdata', FLAGS.channel)
    train_images = data.train.images
    print (train_images.shape)
    train_labels = data.train.labels
    n_z = FLAGS.z_dim
    n_discrete = 0
    n_continuous = 15
    print(n_z)
    
    #-----------------------------------------------------------------------
    #Define placeholders for the iterator
    x = tf.placeholder(tf.float32, shape=[None, data.train.dim1, data.train.dim2,
                                          data.train.num_channels])
    y = tf.placeholder(data.train.labels.dtype, shape=[None, data.train.num_labels])
    batch_size = tf.placeholder(tf.int64, shape=[])
    n_batches = int(data.train.num_examples/FLAGS.batch_size)
    print (data.train.num_examples)
    #Create the dataset and and the iterator that can iterate through it
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=40000)
    dataset = dataset.batch(batch_size)
    data_iter = dataset.make_initializable_iterator()
    features, labels = data_iter.get_next() #Get next elements

    codes = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, n_discrete + n_continuous])
    #-----------------------------------------------------------------------
    #Create the model

    model = Model(learning_rate=FLAGS.learning_rate, input_dim=data.train.dim1,
                  no_channels=data.train.num_channels, z_dim=n_z, infogan=True,
                  continuous_dim=n_continuous, discrete_dim=n_discrete)

    generated_images = model.generator(b_size=FLAGS.batch_size, code=codes)#code=codes)
    D_train_op, G_train_op, real_loss, fake_loss, G_loss, I_loss = model.optimize_network(generated_images,
                                                                                          features)
    #-----------------------------------------------------------------------
    #Summaries for TensorBoard

    train_merge = tf.summary.merge_all()

    generate_placeholder = tf.placeholder(tf.float32, shape=[1, data.train.dim1*10,
                                                             data.train.dim1*10, 1])
    # generate_placeholder_second = tf.placeholder(tf.float32, shape=[1, data.train.dim1*10,
    #                                                                 data.train.dim1*10, 1])
    original_placeholder = tf.placeholder(tf.float32, shape=[1, data.train.dim1*10,
                                                             data.train.dim1*10, 1])
    # original_placeholder_second = tf.placeholder(tf.float32, shape=[1, data.train.dim1*10,
    #                                                                 data.train.dim1*10, 1])

    generate_summary = tf.summary.image('Generated Images',
                                        generate_placeholder, max_outputs=1)

    # generate_summary_second = tf.summary.image('Generated Images second channel',
    #                                            generate_placeholder_second, max_outputs=1)

    original_summary = tf.summary.image('original images', original_placeholder)
    # original_summary_second = tf.summary.image('original images second',
    #                                            original_placeholder_second)

    imgs_summary = tf.summary.merge([generate_summary, original_summary])

    #-----------------------------------------------------------------------
    #Train the model
    var_init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print ('Started Training')
        sess.run(var_init)
        epoch = 1
        train_writer = tf.summary.FileWriter(FLAGS.summary_path+'train_summary', sess.graph)
        image_writer = tf.summary.FileWriter(FLAGS.summary_path+'image_summary', sess.graph)
        batch_counter = 0

        for epoch in range(FLAGS.Epochs):
            start = time.time()
            sess.run(data_iter.initializer, feed_dict={x: train_images,
                                                       batch_size: FLAGS.batch_size,
                                                       y: train_labels})

            fake_loss_total = 0
            real_loss_total = 0
            G_loss_total = 0
            I_loss_total = 0

            size = data.train.num_examples
            code_size = FLAGS.batch_size

            while True:
                if size < 300:
                    break
                try:
                    #discrete_code = np.random.multinomial(1, n_discrete * [float(1.0/n_discrete)],
                    #                                      size=code_size)
                    
                    continuous_code = np.random.uniform(-1, 1, size=(code_size, n_continuous))
                    #all_codes = np.concatenate((discrete_code, continuous_code), axis = 1)
                    all_codes = continuous_code

                    _, rl, fl, il = sess.run([D_train_op, real_loss, fake_loss, I_loss],
                                             feed_dict={codes: all_codes})
                    real_loss_total += rl
                    fake_loss_total += fl
                    I_loss_total += il

                    _, rl, fl, il = sess.run([D_train_op, real_loss, fake_loss, I_loss],
                        feed_dict={codes: all_codes})

                    real_loss_total += rl
                    fake_loss_total += fl
                    I_loss_total += il

                    #train generator
                    _, gl = sess.run([G_train_op, G_loss], feed_dict={codes: all_codes})
                    G_loss_total += gl

                    size -= 3*FLAGS.batch_size

                    if batch_counter % 50 == 0:
                        summary = sess.run(train_merge, feed_dict={codes: all_codes})
                        train_writer.add_summary(summary, batch_counter)
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    break
            end = time.time()
            print('time')
            print (end-start)
            print ('---')
            print (np.random.uniform(-1,1, size=(1,5)))
            print ('---')


            print("Epoch: {}, G loss: {:.4f}, Fake D loss: {:.4f}, Real D los {:.4f}, I loss {:.4f}".format(epoch+1, G_loss_total, fake_loss_total/2, real_loss_total/2, I_loss_total/2))

        saver.save(sess, FLAGS.save_path)
    #-----------------------------------------------------------------------
    # #   #Visualise results in Tensorboard
        sess.run(data_iter.initializer, feed_dict={x: data.train.images.astype('float32'),
                                                   batch_size: 1000, y: data.train.labels})

        #discrete_code = np.random.multinomial(1, n_discrete * [float(1.0/n_discrete)],
        #                                      size=code_size)
        continuous_code = np.random.uniform(-1, 1, size=(code_size, n_continuous))
    #     #all_codes = np.concatenate((discrete_code, continuous_code), axis = 1)
        all_codes = continuous_code

        generated, feat = sess.run([generated_images, features], feed_dict={codes: all_codes})

        sprite_generated = plots.create_sprite_image(generated[:100, :, :])
        # sprite_generated_second = plots.create_sprite_image(generated[:100, :, :, 1])
        sprite_orig = plots.create_sprite_image(feat[:100, :, :])
        sprite_orig_second = plots.create_sprite_image(feat[:100, :, :, 1])
        summary = sess.run(imgs_summary, feed_dict={generate_placeholder: sprite_generated,
                                                    original_placeholder: sprite_orig})
        image_writer.add_summary(summary, epoch)
        #-----------------------------------------------------------------------

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 100, 'size of training batches')
    tf.app.flags.DEFINE_integer('Epochs', 150, 'number of training iterations')
    tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
    tf.app.flags.DEFINE_integer('z_dim', 10, 'Dimension of latent space')
    tf.app.flags.DEFINE_string('channel', 'second', 'Which channel to use, first, second or both')
    tf.app.flags.DEFINE_string('summary_path', './infogan_summaries_morez', 'Path to save summary files')
    tf.app.flags.DEFINE_string('save_path', './infogan_model_morez/model.ckpt', 'path to saved model')

    tf.app.run()