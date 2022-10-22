"""Data Loader"""
import jsonschema
import tensorflow as tf
import tensorflow_datasets as tfds

from configs.data_schema import SCHEMA


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        return tfds.load(data_config.path, with_info=data_config.load_with_info)
    
    @staticmethod
    def preprocess_data(dataset, batch_size, buffer_size,image_size):
        """ Preprocess and splits into training and test"""
        train = dataset['train'].map(lambda image: DataLoader._preprocess_train(image,image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train.shuffle(buffer_size)

        test = dataset['test'].map(lambda image: DataLoader._preprocess_test(image, image_size))
        test_dataset = test.shuffle(buffer_size)

        return train_dataset, test_dataset

    @staticmethod
    def _preprocess_train(datapoint, image_size):
        """ Loads and preprocess  a single training image """
        input_image = tf.image.resize(datapoint['image'], (image_size, image_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (image_size, image_size))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = DataLoader._normalize(input_image, input_mask)

        return input_image, input_mask
    
    @staticmethod
    def _preprocess_test(datapoint, image_size):
        """ Loads and preprocess a single test images """

        input_image = tf.image.resize(datapoint['image'], (image_size, image_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (image_size, image_size))

        input_image, input_mask = DataLoader._normalize(input_image, input_mask)

        return input_image, input_mask

    @staticmethod
    def _normalize(input_image, input_mask):
        """ Normalise input image
        Args:
            input_image (tf.image): The input image
            input_mask (int): The image mask
        Returns:
            input_image (tf.image): The normalized input image
            input_mask (int): The new image mask
        """
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    @staticmethod
    def validate_schema(data_point):
        jsonschema.validate({'image':data_point.tolist()}, SCHEMA)