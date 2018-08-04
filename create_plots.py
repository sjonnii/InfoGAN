import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import io
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import math
from PIL import Image
import tensorflow as tf
import cv2
from tensorflow.contrib.tensorboard.plugins import projector


def plot_confusion_matrix(cm, classes, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes, rotation=90)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.savefig('confmat.png')
    img = Image.open("confmat.png")
    arr = np.array(img)
    arr = arr.reshape([1, 480, 640, 4])
    return arr

def plot_roc(logit, labels):
    plt.figure()
    fpr, tpr, thresholds = roc_curve(labels, logit)
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    #plt.plot([0, 1], [0, 1], 'r-')  # random predictions curve
    x = np.linspace(0,1, len(fpr))
    plt.plot(fpr, (1-fpr), 'k--')
    idx = np.argwhere(np.diff(np.sign(tpr - (1-fpr))) != 0).reshape(-1) + 0
    plt.plot(fpr[idx], tpr[idx], 'ro')

    bbox_args = dict(boxstyle="round", fc="0.8")
    arrow_args = dict(arrowstyle="->")

    # Here we'll demonstrate the extents of the coordinate system and how
    # we place annotating text.

    print (fpr[idx][0])

    plt.annotate('FPR:{:.2f} \n TPR:{:.2f}  \n Threshold:{:.2f}'.format(fpr[idx][0], tpr[idx][0], 
                thresholds[idx][0]),
                xy=(fpr[idx][0], tpr[idx][0]),
                xytext=(fpr[idx][0]+0.3, tpr[idx][0]-0.15),
                ha="left", va="bottom",
                bbox=bbox_args,
                arrowprops=arrow_args)


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic', fontsize=20)
    plt.legend(loc="lower right")
    plt.savefig('roc.png')
    img = Image.open('roc.png')
    arr = np.array(img)
    arr = arr.reshape([1, 480, 640, 4])
    return arr

def high_low_conf(logit, labels, feat, prediction):

    incorr_preds = np.not_equal(labels, prediction).astype('float32')
    prob0 = logit[:, 0]
    prob1 = logit[:, 1]
    diff = np.abs(np.subtract(prob0, prob1))
    incorr_diff = np.multiply(incorr_preds, diff)
    ind_conf_inc = np.argsort(incorr_diff)[-50:]
    ind_lowconf = np.argsort(diff)[0:50]

    features_conf_inc = feat[ind_conf_inc, :, :, 0]
    features_conf_inc = np.reshape(features_conf_inc, [50, 64, 64, 1])
    features_lowconf = feat[ind_lowconf, :, :, 0]
    features_lowconf = np.reshape(features_lowconf, [50, 64, 64, 1])

    labels_lowconf = labels[ind_lowconf]
    pred_lowconf = prediction[ind_lowconf]
    labels_conf_inc = labels[ind_conf_inc]
    pred_conf_inc = prediction[ind_conf_inc]

    p0_lowconf = prob0[ind_lowconf]
    p1_lowconf = prob1[ind_lowconf]
    p0_conf_inc = prob0[ind_conf_inc]
    p1_conf_inc = prob1[ind_conf_inc]

    font                   = cv2.FONT_HERSHEY_PLAIN
    bottomLeftCornerOfText = (1, 5)
    bottomLeftCornerOfText2 = (1, 60)
    fontScale              = 0.5
    fontColor              = (2, 2, 2)
    lineType               = 1

    for i in range(50):
        if labels_lowconf[i] == 0:
            conf = '{:.4f}'.format(p1_lowconf[i])
        elif labels_lowconf[i] == 1:
            conf = '{:.4}'.format(p0_lowconf[i])

        text = 'Label:'+str(labels_lowconf[i])+' pred:' + str(pred_lowconf[i])

        features_lowconf[i] = cv2.putText(features_lowconf[i], text,
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        text = str(conf)
        features_lowconf[i] = cv2.putText(features_lowconf[i],text, 
            bottomLeftCornerOfText2, 
            font, 
            fontScale,
            fontColor,
            lineType)

        if labels_conf_inc[i] == 0:
            conf = '{:.4f}'.format(p1_conf_inc[i])
        elif labels_conf_inc[i] == 1:
            conf = '{:.4f}'.format(p0_conf_inc[i])

        text = 'Label:'+str(labels_conf_inc[i])+' pred:' + str(pred_conf_inc[i])
        features_conf_inc[i] = cv2.putText(features_conf_inc[i],text, 
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)
        text = str(conf)
        features_conf_inc[i] = cv2.putText(features_conf_inc[i],text,
            bottomLeftCornerOfText2,
            font,
            fontScale,
            fontColor,
            lineType)
    
    return features_conf_inc, features_lowconf


def plot_embeddings(embedding, onehot_labels, path):

    #sess = tf.InteractiveSession()

    with tf.device("/cpu:0"):
        tf_embedding = tf.Variable(embedding, trainable=False, name = "embedding")

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    #path = "tensorboard_embeddings"
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(path, sess.graph)
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = "embedding"
    embed.metadata_path = "metadata.tsv"
    projector.visualize_embeddings(writer, config)
    saver.save(sess, path+'/model.ckpt' , global_step=10000)

    with open(path+'metadata.tsv', 'w') as f:
        for i in range(embedding.shape[0]):
            c = np.nonzero(onehot_labels[::1])[1:][0][i]
            f.write('{}\n'.format(c))

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    images = images.reshape([images.shape[0], images.shape[1], images.shape[2]])
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    plt.figure()
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img

    plt.savefig('sprite_image')

    spriteimage = spriteimage.reshape([1, spriteimage.shape[0], spriteimage.shape[1], 1])

    plt.close()
    
    return spriteimage