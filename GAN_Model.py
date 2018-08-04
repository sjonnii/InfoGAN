import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, learning_rate=0.01, input_dim=64, no_channels=2, z_dim=50, infogan=True,
                 continuous_dim=2, discrete_dim=10):
        self.lr = learning_rate
        self.input_dim = input_dim
        self.no_channels = no_channels
        self.z_dim = z_dim
        self.infogan = infogan
        self.continuous_dim = continuous_dim
        self.discrete_dim = discrete_dim
        self.code_dim = self.continuous_dim + self.discrete_dim
        print(self.continuous_dim)
        print(self.discrete_dim)

    def generator(self, b_size, code=[]):

        with tf.variable_scope('Generator_variables'):

            self.code = code

            z_ = tf.random_normal(tf.stack([b_size, self.z_dim]))
            #z_ = np.random.normal(size=(b_size, self.z_dim))

            if self.infogan:
                z = tf.concat(axis=1, values=[z_, code])
            else:
                z = z_
            
            transform = tf.layers.dense(z, 4 * 4 * 512)

            reshape = tf.reshape(transform, [tf.shape(z)[0], 4, 4, 512])

            #4x4x1024

            print (reshape.shape)

            deconv1 = tf.layers.conv2d_transpose(
                inputs=reshape,
                filters=256,
                kernel_size=[3, 3],
                strides=2,
                padding="same",
                activation=tf.nn.relu,
                )

            deconv1 = tf.layers.batch_normalization(inputs=deconv1)
            print (deconv1.shape)

            #8x8x512

            deconv2 = tf.layers.conv2d_transpose(
                inputs=deconv1,
                filters=128,
                kernel_size=[3, 3],
                strides=2,
                padding="same",
                activation=tf.nn.relu,
                )

            deconv2 = tf.layers.batch_normalization(inputs=deconv2)
            print (deconv2.shape)

            #16x16x256

            deconv3 = tf.layers.conv2d_transpose(
                inputs=deconv2,
                filters=64,
                kernel_size=[3, 3],
                strides=2,
                padding="same",
                activation=tf.nn.relu,
                )

            deconv3 = tf.layers.batch_normalization(inputs=deconv3)
            print (deconv3.shape)

            #32x32x128

            deconv4 = tf.layers.conv2d_transpose(
                inputs=deconv3,
                filters=32,
                kernel_size=[3, 3],
                strides=2,
                padding="same",
                activation=tf.nn.relu,
                )

            deconv4 = tf.layers.batch_normalization(inputs=deconv4)
            print (deconv4.shape)

            #64x64x64

            image = tf.layers.conv2d(
                inputs=deconv4,
                filters=self.no_channels,
                kernel_size=[1, 1],
                strides=1,
                padding="same",
                activation=tf.nn.sigmoid)

            #64x64x1
        print ('image')
        print (image.shape)

        return image

    def discriminator(self, images):

        with tf.variable_scope('discriminator_variables', reuse=tf.AUTO_REUSE):

            #64x64x1
            conv1 = tf.layers.conv2d(
                inputs=images,
                filters=32,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.leaky_relu,
                )

            conv1 = tf.layers.batch_normalization(inputs=conv1)
            #64x64x64
            print (conv1.shape)

            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=[3, 3],
                strides=2,
                padding="same",
                activation=tf.nn.leaky_relu,
                )

            conv2 = tf.layers.batch_normalization(inputs=conv2)
            #32x32x128
            print (conv2.shape)

            conv3 = tf.layers.conv2d(
                inputs=conv2,
                filters=128,
                kernel_size=[3, 3],
                strides=2,
                padding="same",
                activation=tf.nn.leaky_relu,
                )

            conv3 = tf.layers.batch_normalization(inputs=conv3)
            #16x16x256
            print (conv3.shape)

            conv4 = tf.layers.conv2d(
                inputs=conv3,
                filters=256,
                kernel_size=[3, 3],
                strides=2,
                padding="same",
                activation=tf.nn.leaky_relu,
                )

            conv4 = tf.layers.batch_normalization(inputs=conv4)
            #8x8x512
            print (conv4.shape)

            conv5 = tf.layers.conv2d(
                inputs=conv4,
                filters=512,
                kernel_size=[3, 3],
                strides=2,
                padding="same",
                activation=tf.nn.leaky_relu,
                )
            conv5 = tf.layers.batch_normalization(inputs=conv5)
            #4x4x1024
            print(conv5.shape)

            flatten = tf.reshape(conv5, [-1, int(self.input_dim/16) *
                                              int(self.input_dim/16) * 512])

            probability = tf.layers.dense(flatten, 1)

            if self.infogan:
                fc = tf.layers.dense(flatten, 256, activation=tf.nn.relu)
                info_logit = tf.layers.dense(fc, self.code_dim)
            else:
                info_logit = 0

        return probability, info_logit

    def optimize_network(self, generated_images, real_images):

        real_D_output, _ = self.discriminator(real_images)
        fake_D_output, predicted_codes = self.discriminator(generated_images)

        print (real_D_output.shape)
        print ('Real_D_output')
        print (fake_D_output.shape)
        print ('Fake_d_output')
        print (predicted_codes.shape)
        print ('predicted codes')

        #calculate loss of discriminator
        real_D_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                                    multi_class_labels=tf.ones_like(real_D_output),
                                    logits=real_D_output))

        fake_D_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                                     multi_class_labels=tf.zeros_like(real_D_output),
                                     logits=fake_D_output))

        discriminator_loss = real_D_loss + fake_D_loss

        disc_reg = self.Discriminator_Regularizer(real_D_output, real_images,
                                                  fake_D_output, generated_images)
        gamma = 2
        assert discriminator_loss.shape == disc_reg.shape
        discriminator_loss += disc_reg

        generator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                                        multi_class_labels=tf.ones_like(fake_D_output),
                                        logits=fake_D_output))

        if self.infogan:
            discrete_code_pred = predicted_codes[:, :self.discrete_dim]
            discrete_code_target = self.code[:, :self.discrete_dim]
            discrete_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                                           multi_class_labels=discrete_code_target,
                                           logits=discrete_code_pred))
            #continuous loss:
            continuous_code_pred = predicted_codes[:, self.discrete_dim:]
            continuous_code_target = self.code[:, self.discrete_dim:]
            continuous_loss = tf.losses.mean_squared_error(continuous_code_target,
                                                           continuous_code_pred)

            inf_loss = discrete_loss + continuous_loss
        else:
            inf_loss = 0

        discriminator_loss += inf_loss
        generator_loss += inf_loss

        discriminator_variables = tf.trainable_variables(scope='discriminator_variables')
        generator_variables = tf.trainable_variables(scope='Generator_variables')
        discriminator_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        generator_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        D_train_op = discriminator_optimizer.minimize(discriminator_loss,
                                                      var_list=discriminator_variables)
        G_train_op = generator_optimizer.minimize(generator_loss,
                                                  var_list=generator_variables)

        tf.summary.scalar('Discriminator loss - Real data', real_D_loss)
        tf.summary.scalar('Discriminator loss - Fake data', fake_D_loss)
        tf.summary.scalar('generator_loss', generator_loss)
        tf.summary.scalar('Discriminator - Total loss', discriminator_loss)
        tf.summary.scalar('Information Loss - Total loss', inf_loss)

        return D_train_op, G_train_op, real_D_loss, fake_D_loss, generator_loss, inf_loss

    def Discriminator_Regularizer(self, D1_logits, D1_arg, D2_logits, D2_arg):
        D1 = tf.nn.sigmoid(D1_logits)
        D2 = tf.nn.sigmoid(D2_logits)
        grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
        grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
        grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, [tf.shape(D1_logits)[0], -1]),
                                      axis=1, keep_dims=True)
        grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, [tf.shape(D1_logits)[0], -1]),
                                      axis=1, keep_dims=True)

        #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
        tf.assert_equal(tf.shape(grad_D1_logits_norm), tf.shape(D1))
        tf.assert_equal(tf.shape(grad_D2_logits_norm), tf.shape(D2))
        print(grad_D1_logits_norm.shape)
        print(D1.shape)

        reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
        reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
        disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
        return disc_regularizer

    def metrics():
        pass