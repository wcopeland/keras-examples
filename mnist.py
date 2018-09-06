import json
import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from clusterone import get_data_path, get_logs_path

from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

tf.logging.set_verbosity(tf.logging.INFO)
try:
    job_name = os.environ['JOB_NAME']
    task_index = int(os.environ['TASK_INDEX'])
    ps_hosts = os.environ['PS_HOSTS'].split(',')
    worker_hosts = os.environ['WORKER_HOSTS'].split(',')
    TF_CONFIG = {'task': {'type': job_name, 'index': task_index},
                 'cluster': {'chief': [worker_hosts[0]],
                             'worker': worker_hosts,
                             'ps': ps_hosts},
                 'environment': 'cloud'}
    local_ip = 'localhost:' + TF_CONFIG['cluster'][job_name][task_index].split(':')[1]
    if (job_name == 'chief') or (job_name == 'worker' and task_index == 0):
        job_name = 'chief'
        TF_CONFIG['task']['type'] = 'chief'
        TF_CONFIG['cluster']['worker'][0] = local_ip
    TF_CONFIG['cluster'][job_name][task_index] = local_ip
    os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)
except KeyError as ex:
    pass

def get_args():
    '''Return parsed args'''
    parser = ArgumentParser()
    parser.add_argument('--local_data_dir', type=str, default='data/',
                        help='Path to local data directory')
    parser.add_argument('--local_log_dir', type=str, default='logs/',
                        help='Path to local log directory')
    # Model params
    parser.add_argument('--hidden_units', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function. See Keras activation functions. Default: relu')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    opts = parser.parse_args()

    opts.data_dir = get_data_path(dataset_name = 'adrianyi/mnist-data',
                                 local_root = opts.local_data_dir,
                                 local_repo = '',
                                 path = '')
    opts.log_dir = get_logs_path(root = opts.local_log_dir)

    return opts

def get_model(opts):
    '''Return Keras model'''
    input_tensor = tf.keras.layers.Input(shape=(784,), name='input')

    temp = input_tensor
    for i, n_units in enumerate(opts.hidden_units):
        temp = tf.keras.layers.Dense(n_units, activation=opts.activation, name='fc'+str(i))(temp)
    output = tf.keras.layers.Dense(10, activation='softmax', name='output')(temp)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
    optimizer = tf.keras.optimizers.Adam(lr=opts.learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

class IteratorInitializerHook(tf.train.SessionRunHook):
    '''From https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0'''
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)

def get_inputs(x, y, batch_size=64, train=True):
    '''Returns input function and and iterator initializer hook'''
    iterator_initializer_hook = IteratorInitializerHook()

    def input_fn():
        '''Input function to be used for Estimator class'''
        x_placeholder = tf.placeholder(tf.float32, x.shape)
        y_placeholder = tf.placeholder(tf.float32, y.shape)
        data_x = tf.data.Dataset.from_tensor_slices(x_placeholder)
        data_y = tf.data.Dataset.from_tensor_slices(y_placeholder)
        data = tf.data.Dataset.zip((data_x, data_y))
        data = data.batch(batch_size)
        if train:
            data = data.repeat(count=None).shuffle(buffer_size=5*batch_size)
        else:
            data = data.repeat(count=1)

        data = data.prefetch(8)

        iterator = data.make_initializable_iterator()
        next_example, next_label = iterator.get_next()

        # Set runhook to initialize iterator
        iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(
            iterator.initializer,
            feed_dict={x_placeholder: x, y_placeholder: y}
        )

        return next_example, next_label

    return input_fn, iterator_initializer_hook

def main(opts):
    '''Main function'''
    data = read_data_sets(opts.data_dir,
                          one_hot=False,
                          fake_data=False)

    model = get_model(opts)
    config = tf.estimator.RunConfig(
                model_dir=opts.log_dir,
                save_summary_steps=500,
                save_checkpoints_steps=500,
                keep_checkpoint_max=5,
                log_step_count_steps=10)
    classifier = tf.keras.estimator.model_to_estimator(model, model_dir=opts.log_dir, config=config)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'input': data.train.images},
                         y=tf.keras.utils.to_categorical(data.train.labels.astype(np.int32), 10).astype(np.float32),
                         num_epochs=None,
                         batch_size=opts.batch_size,
                         shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'input': data.test.images},
                         y=tf.keras.utils.to_categorical(data.test.labels.astype(np.int32), 10).astype(np.float32),
                         num_epochs=1,
                         shuffle=False)

    train_input_fn, train_iter_hook = get_inputs(data.train.images,
        tf.keras.utils.to_categorical(data.train.labels.astype(np.int32), 10).astype(np.float32),
        batch_size=opts.batch_size,
        train=True)
    eval_input_fn, eval_iter_hook = get_inputs(data.test.images,
        tf.keras.utils.to_categorical(data.test.labels.astype(np.int32), 10).astype(np.float32),
        batch_size=opts.batch_size,
        train=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=1e6,
                                        hooks=[train_iter_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=60,
                                      hooks=[eval_iter_hook])

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

if __name__ == '__main__':
    opts = get_args()
    main(opts)
