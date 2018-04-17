#  Copyright 2018 Indiana University Southeast. All Rights Reserved.
#  Author: Jared Sanders

# MIT License
#
# Copyright (c) [year] [fullname]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import load_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1000, type=int, help='batch size')
parser.add_argument('--train_steps', default=10000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = load_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 4 classes.
        n_classes=6)

    # Train the Model.
    classifier.train(
        input_fn=lambda:load_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:load_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # 1	1	1	1	1	1	6	1	1	2	1	3	1	77.75
    # 1	1	1	4	1	1	3	1	1	1	1	1	1	45.465
    # Generate predictions from the model
    expected = ["C", "C", "C", "C", "F"]
    predict_x = {
        't0': [1, 2, 2, 1, 1],
        't1': [1, 3, 1, 1, 1],
        't2': [1, 3, 2, 1, 1],
        't3': [2, 3, 1, 1, 4],
        't4': [2, 2, 1, 1, 1],
        't5': [2, 2, 1, 1, 1],
        't6': [2, 17,6, 6, 3],
        't7': [1, 2, 2, 1, 1],
        't8': [3, 6, 2, 1, 1],
        't9': [1, 9, 3, 2, 1],
        't10':[8, 13,9, 1, 1],
        't11':[1, 4, 1, 3, 1],
        't12':[8, 3, 2, 1, 1]

    }

    predictions = classifier.predict(
        input_fn=lambda:load_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(load_data.GRADES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
