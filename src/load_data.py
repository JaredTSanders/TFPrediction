import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://18.221.149.34/files/csv/tensorflow/train.csv"
TEST_URL = "http://18.221.149.34/files/csv/tensorflow/test.csv" # These no longer work, AWS was charging way too much. I have uploaded the files in question to the resources folder

# The data I was given is inconsistent for the number of tests, so this may change drastically. 
CSV_COLUMN_NAMES = ['t0', 't1',
                    't2', 't3', 't4', 't5', 't6','t7','t8','t9','t10','t11', 't12', 'Grades']
GRADES = ['F', 'D', 'C', 'B', 'A']

# Would actually download the files from my server if it was still on. Need to change to local
def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

# Parsing the CSV
def load_data(y_name='Grades'):
    """Returns the dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

# Training steps
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

# Takes input and evaluates
def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


# The below came mostly from Google's example. This does the initial reading of the CSV, the parser above simply puts it in (train_x, train_y) format
def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Grades')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
