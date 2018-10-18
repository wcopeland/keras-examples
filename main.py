import json
import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from clusterone import get_data_path, get_logs_path

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.logging.set_verbosity(tf.logging.INFO)
try:
    # Create necessary TF_CONFIG environment variable for Estimator
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
    parser.add_argument('--hidden_units', type=int, nargs='*', default=[32, 64])
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=512)
    opts = parser.parse_args()

    opts.data_dir = get_data_path(dataset_name = '*/*',
                                 local_root = opts.local_data_dir,
                                 local_repo = '',
                                 path = '')
    opts.log_dir = get_logs_path(root = opts.local_log_dir)

    return opts

def get_model(opts):
    '''Return a CNN Keras model'''
    input_tensor = tf.keras.layers.Input(shape=(784,), name='input')

    temp = tf.keras.layers.Reshape([28, 28, 1], name='input_image')(input_tensor)
    for i, n_units in enumerate(opts.hidden_units):
        temp = tf.keras.layers.Conv2D(n_units, kernel_size=3, strides=(2, 2),
                                      activation='relu', name='cnn'+str(i))(temp)
        temp = tf.keras.layers.Dropout(opts.dropout, name='dropout'+str(i))(temp)
    temp = tf.keras.layers.GlobalAvgPool2D(name='average')(temp)
    output = tf.keras.layers.Dense(10, activation='softmax', name='output')(temp)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
    optimizer = tf.keras.optimizers.Adam(lr=opts.learning_rate, decay=opts.learning_decay)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def main(opts):
    '''Main function'''
    data = read_data_sets(opts.data_dir,
                          one_hot=False,
                          fake_data=False)

    model = get_model(opts)
    config = tf.estimator.RunConfig(
                model_dir=opts.log_dir,
                save_summary_steps=1,
                save_checkpoints_steps=100,
                keep_checkpoint_max=3,
                log_step_count_steps=10)
    classifier = tf.keras.estimator.model_to_estimator(model, model_dir=opts.log_dir, config=config)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'input': data.train.images},
                         y=tf.keras.utils.to_categorical(data.train.labels.astype(np.int32), 10).astype(np.float32),
                         num_epochs=None,
                         batch_size=opts.batch_size,
                         shuffle=True,
                         queue_capacity=10*opts.batch_size,
                         num_threads=4)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'input': data.test.images},
                         y=tf.keras.utils.to_categorical(data.test.labels.astype(np.int32), 10).astype(np.float32),
                         num_epochs=1,
                         shuffle=False,
                         queue_capacity=10*opts.batch_size,
                         num_threads=4)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=1e6)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=60)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

if __name__ == '__main__':
    opts = get_args()
    main(opts)
