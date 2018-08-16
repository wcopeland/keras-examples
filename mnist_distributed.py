"""
Example of how to train a basic Keras model on MNIST on the ClusterOne platform.

This example leverages the work of fchollet and vpj:
https://gist.github.com/fchollet/2c9b029f505d94e6b8cd7f8a5e244a4e
https://gist.github.com/vpj/e03c32819641dd65e0e70e563a56be42
"""
import tensorflow as tf
import keras
import os
from argparse import ArgumentParser

#
# Command line arguments
#
parser = ArgumentParser()
parser.add_argument('--local_data_dir', type=str, default='data/',
                        help='Path to local data directory')
parser.add_argument('--local_log_dir', type=str, default='logs/',
                        help='Path to local log directory')
args = parser.parse_args()


# ----- Insert that snippet to run distributed jobs -----

from clusterone import get_data_path, get_logs_path

# Specifying paths when working locally
# For convenience we use a clusterone wrapper (get_data_path below) to be able
# to switch from local to clusterone without cahnging the code.

PATH_TO_LOCAL_LOGS = args.local_log_dir
ROOT_PATH_TO_LOCAL_DATA = args.local_data_dir 

# Configure  distributed task
try:
  job_name = os.environ['JOB_NAME']
  task_index = os.environ['TASK_INDEX']
  ps_hosts = os.environ['PS_HOSTS']
  worker_hosts = os.environ['WORKER_HOSTS']
except:
  job_name = None
  task_index = 0
  ps_hosts = None
  worker_hosts = None

flags = tf.app.flags

# Flags for configuring the distributed task
flags.DEFINE_string("job_name", job_name,
                    "job name: worker or ps")
flags.DEFINE_integer("task_index", task_index,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the chief worker task that performs the variable "
                     "initialization and checkpoint handling")
flags.DEFINE_string("ps_hosts", ps_hosts,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", worker_hosts,
                    "Comma-separated list of hostname:port pairs")

# Training related flags
flags.DEFINE_string("data_dir",
                    get_data_path(
                        dataset_name = "malo/mnist", #all mounted repo
                        local_root = ROOT_PATH_TO_LOCAL_DATA,
                        local_repo = "mnist",
                        path = 'data'
                        ),
                    "Path to store logs and checkpoints. It is recommended"
                    "to use get_logs_path() to define your logs directory."
                    "so that you can switch from local to clusterone without"
                    "changing your code."
                    "If you set your logs directory manually make sure"
                    "to use /logs/ when running on ClusterOne cloud.")
flags.DEFINE_string("log_dir",
                    ROOT_PATH_TO_LOCAL_DATA,
                    "Path to dataset. It is recommended to use get_data_path()"
                    "to define your data directory.so that you can switch "
                    "from local to clusterone without changing your code."
                    "If you set the data directory manually makue sure to use"
                    "/data/ as root path when running on ClusterOne cloud.")
FLAGS = flags.FLAGS

def device_and_target():
    # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
    # Don't set a device.
    if FLAGS.job_name is None:
        print("Running single-machine training")
        return (None, "")

    # Otherwise we're running distributed TensorFlow
    print("Running distributed training")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
        raise ValueError("Must specify an explicit `ps_hosts`")
    if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
        raise ValueError("Must specify an explicit `worker_hosts`")

    cluster_spec = tf.train.ClusterSpec({
        "ps": FLAGS.ps_hosts.split(","),
        "worker": FLAGS.worker_hosts.split(","),
    })
    server = tf.train.Server(
        cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()

    worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    return (
        tf.train.replica_device_setter(
        worker_device=worker_device,
        cluster=cluster_spec),
        server.target,
)

# --- end of snippet ----

def main(_):

    #
    # Data
    #
    mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=FLAGS.data_dir, one_hot=True)

    device, target = device_and_target()  # getting node environment
    with tf.device(device):
        # set Keras learning phase to train
        keras.backend.set_learning_phase(1)
        # do not initialize variables on the fly
        keras.backend.manual_variable_initialization(True)

        #
        # Keras model
        #
        model_inp = keras.layers.Input(shape=(784,))
        x = keras.layers.Dense(128, activation='relu')(model_inp)
        x = keras.layers.Dense(128, activation='relu')(x)
        model_out = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.models.Model(model_inp, model_out)

        # keras model predictions
        predictions = model.output
        # placeholder for training targets
        targets = tf.placeholder(tf.float32, shape=(None, 10))
        # categorical crossentropy loss
        loss = tf.reduce_mean(
            keras.losses.categorical_crossentropy(targets, predictions)
        )
        # accuracy
        acc_value = tf.reduce_mean(keras.metrics.categorical_accuracy(targets, predictions))
        # global step
        global_step = tf.train.get_or_create_global_step()

        # Only if you have regularizers, not in this example
        total_loss = loss * 1.0  # Copy
        for regularizer_loss in model.losses:
            tf.assign_add(total_loss, regularizer_loss)

        optimizer = tf.train.AdamOptimizer()

        # Barrier to compute gradients after updating moving avg of batch norm
        with tf.control_dependencies(model.updates):
            barrier = tf.no_op(name="update_barrier")

        with tf.control_dependencies([barrier]):
            grads = optimizer.compute_gradients(
                total_loss,
                model.trainable_weights
            )
            grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

        with tf.control_dependencies([grad_updates]):
            train_op = tf.identity(total_loss, name="train")

    #
    # Train
    #
    with tf.train.MonitoredTrainingSession(
            master=target, is_chief=(FLAGS.task_index == 0), checkpoint_dir=FLAGS.log_dir) as sess:

        for i in range(1000):
            batch_x, batch_y = mnist.train.next_batch(128)

            # perform the operations we defined earlier on batch
            loss_value = sess.run(
                [train_op],
                feed_dict = {
                    model.inputs[0]: batch_x,
                    targets: batch_y
                }
            )

            val_acc = sess.run(
                acc_value,
                feed_dict={
                    model.inputs[0]: mnist.test.images,
                    targets: mnist.test.labels
                }
            )

            print('Batch Number: {0:4d}, Task: {1:3d}, Validation Accuracy: {2:6.4f}'.format(i, FLAGS.task_index, val_acc))


if __name__ == '__main__':
    tf.app.run()
