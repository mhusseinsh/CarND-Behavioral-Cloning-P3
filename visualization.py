import os
import visualkeras
from keras.utils.vis_utils import plot_model
from collections import defaultdict
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D
import matplotlib.pyplot as plt
import errno


def check_dir(filename):
    # check dir: checks the path of a given filename/directory, if it doesn't exist, then create the path
    #
    # filename given filename/directory to be checked
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

class Visualization():
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir

    def visualize_model(self):
        color_map = defaultdict(dict)
        color_map[Convolution2D]['fill'] = 'orange'
        color_map[Lambda]['fill'] = 'gray'
        color_map[Dropout]['fill'] = 'pink'
        color_map[Cropping2D]['fill'] = 'red'
        color_map[Dense]['fill'] = 'green'
        color_map[Flatten]['fill'] = 'blue'
        color_map[Activation]['fill'] = 'yellow'
        file_name = os.path.join(self.output_dir, "color_output.png")
        check_dir(file_name)
        visualkeras.layered_view(self.model, color_map=color_map, spacing=50, to_file=file_name)
        file_name = os.path.join(self.output_dir, "output.png")
        check_dir(file_name)
        visualkeras.layered_view(self.model, spacing=50, to_file=file_name)  # write to disk
        file_name = os.path.join(self.output_dir, "flat_output.png")
        check_dir(file_name)
        visualkeras.layered_view(self.model, draw_volume=False, spacing=50, to_file=file_name)
        file_name = os.path.join(self.output_dir, "model_plot.png")
        check_dir(file_name)
        plot_model(self.model, to_file=file_name, show_shapes=True, show_layer_names=True)

    def save_plots(self, results):
        # Plot the training and validation loss for each epoch
        plt.plot(results.history['loss'])
        plt.plot(results.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        file_name = os.path.join(self.output_dir, "model_history.png")
        check_dir(file_name)
        plt.savefig(file_name)
