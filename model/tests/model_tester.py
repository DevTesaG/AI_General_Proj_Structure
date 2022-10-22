
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from model.unet import UNet
from configs.config import CFG
from unittest.mock import patch


# import unittest


# class Tester(unittest.TestCase): This one is for none tenserflow solutions

def dummy_load_data(*args, **kwargs):
    with tfds.testing.mock_data(num_examples=1):
        return tfds.load(CFG['data']['path'], with_info=True)


class Tester(tf.test.TestCase):
    """Tester model class"""
    def __init__(self, model) -> None:
        self.model2Test = model
    
    def setUP(self):
        super(Tester, self).setUp()
        self.unet = UNet(CFG)

    def tearDown(self):
        pass
    
    def test_normalize(self):
        input_image = np.array([[1., 1.], [1., 1.]])
        input_mask = 1
        expected_image = np.array([[0.00392157, 0.00392157], [0.00392157, 0.00392157]])

        result = self.model2Test._normalize(input_image, input_mask)
        self.assertEquals(expected_image, result[0])

    """Test the model architecture"""
    def test_ouput_size(self):
        shape = (1, self.unet.image_size, self.unet.image_size, 3)
        image = tf.ones(shape) # construct dummy input
        self.unet.build()
        self.assertEqual(self.unet.model.predict(image).shape, shape)

    @patch('model.unet.DataLoader.load_data')
    def test_load_data(self, mock_data_loader):
        mock_data_loader.side_effect = dummy_load_data
        shape = tf.TensorShape([None, self.unet.image_size, self.unet.image_size, 3])

        self.unet.load_data()
        mock_data_loader.assert_called()

        self.assertItemsEqual(self.unet.train_dataset.element_spec[0].shape, shape)
        self.assertItemsEqual(self.unet.test_dataset.element_spec[0].shape, shape)


if __name__ == '__main__':
    tf.test.main()