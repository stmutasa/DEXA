# Defines and builds the localization network
#    Computes input images and labels using inputs()
#    Computes inference on the models (forward pass) using inference()
#    Computes the total loss using loss()
#    Performs the backprop using train()

_author_ = 'simi'

import tensorflow as tf
import Input as Input
import SODNetwork as SDN

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Retreive helper function object
sdn = SDN.SODMatrix()
sdloss = SDN.SODLoss(2)


def inputs(training=True, skip=False):

    """
    Loads the inputs
    :param filenames: Filenames placeholder
    :param training: if training phase
    :param skip: Skip generating tfrecords if already done
    :return:
    """

    if not skip:  Input.pre_proc_localizations(FLAGS.box_dims, thresh=0.6)

    else: print('-------------------------Previously saved records found! Loading...')

    return Input.load_protobuf_loc(training)


def forward_pass_RPN(images, phase_train):

    """
    Train a 2 dimensional network. Default input size 64x64
    :param images: self explanatory
    :param phase_train: True if this is the training phase
    :return: L2 Loss and Logits
    """

    # Initial kernel size
    K = 8
    #images = tf.expand_dims(images, -1)
    images = tf.cast(images, tf.float32)

    # Inputs = batchx64x64x64
    conv = sdn.residual_layer('Conv1', images, 3, K, S=1, phase_train=phase_train) # 64
    conv = sdn.inception_layer('Conv1ds', conv, K * 2, S=2, phase_train=phase_train) # 32

    conv = sdn.residual_layer('Conv2a', conv, 3, K * 2, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv2b', conv, 3, K * 2, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv2ds', conv, K * 4, S=2, phase_train=phase_train)  # 16

    conv = sdn.residual_layer('Conv3a', conv, 3, K * 4, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv3b', conv, 3, K * 4, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv3c', conv, 3, K * 4, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv3ds', conv, K * 8, S=2, phase_train=phase_train)  # 8

    conv = sdn.residual_layer('Conv4a', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Conv4b', conv, K * 8, S=1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4c', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4d', conv, 3, K * 8, 1, phase_train=phase_train) # Smaller net here
    conv = sdn.residual_layer('Conv4e', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4f', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4g', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Conv4h', conv, 3, K * 8, 1, phase_train=phase_train) # Smaller End Here

    convC = sdn.inception_layer('ConvC1', conv, K * 16, S=2, phase_train=phase_train)  # 4
    convC = sdn.residual_layer('ConvC2', convC, 3, K * 16, 1, phase_train=phase_train)
    convC = sdn.residual_layer('ConvC3', convC, 3, K * 16, 1, phase_train=phase_train)
    convC = sdn.residual_layer('ConvC4', convC, 3, K * 16, 1, phase_train=phase_train)
    linearC = sdn.fc7_layer('FC7c', convC, 8, True, phase_train, FLAGS.dropout_factor, override=3, BN=True)
    linearC = sdn.linear_layer('LinearC', linearC, 4, True, phase_train, FLAGS.dropout_factor, BN=True)
    LogitsC = sdn.linear_layer('SoftmaxC', linearC, 2, relu=False, add_bias=False, BN=False)

    # Return logits
    return LogitsC


def total_loss(logits, labels):

    """
    Add loss to the trainable variables and a summary
    logits is a tuple of Classification logits and regression logits
    Saved the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
        10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]
    """

    # Loss factors: 1e2, 0.0 and 2.0 lead to nan. 1e2, 1e-3 and 1.0 lead to nana
    class_loss_factor = 1.0

    # Squish
    labels = tf.cast(tf.squeeze(labels), tf.float32)
    logitsC = tf.squeeze(logits)

    """
    Classification loss
    """

    # Change labels to one hot
    labelsC = tf.one_hot(tf.cast(labels[:, 19], tf.uint8), depth=2, dtype=tf.uint8)

    # Use focal loss
    class_loss = tf.reduce_sum(sdloss.focal_softmax_cross_entropy_with_logits(labelsC, logitsC, alpha=[0.5, 0.5]))

    # Normalize by minibatch size
    class_loss = class_loss_factor*tf.divide(class_loss, FLAGS.batch_size)

    # Output the summary of the MSE and MAE
    tf.summary.scalar('Class_Loss', class_loss)

    # Add loss to the collection
    tf.add_to_collection('losses', class_loss)

    return class_loss


def backward_pass(total_loss):

    """
    This function performs our backward pass and updates our gradients
    """

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Decay the learning rate and use cosine annealing
    dk_steps = int((FLAGS.epoch_size / FLAGS.batch_size) * 21)
    lr_decayed = tf.train.cosine_decay_restarts(FLAGS.learning_rate, global_step, dk_steps)

    # Compute the gradients. NAdam optimizer
    opt = tf.contrib.opt.NadamOptimizer(learning_rate=lr_decayed, beta1=FLAGS.beta1,
                                        beta2=FLAGS.beta2, epsilon=0.1)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # clip the gradients
    #gradients = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gradients]

    # Apply the gradients
    train_op = opt.apply_gradients(gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([train_op, variable_averages_op]):  # Wait until we apply the gradients
        dummy_op = tf.no_op(name='train')  # Does nothing. placeholder to control the execution of the graph

    return dummy_op