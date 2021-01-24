import csv
import os
import sklearn
from sklearn.model_selection import train_test_split
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from sklearn.utils import shuffle
import subprocess


class PreprocessData:
    def __init__(self, data_path, labels_file, correction_factor, skip_header, use_side_images, train_valid_split,
                 do_flip):
        self.data_path = data_path
        self.labels_file = labels_file
        self.skip_header = skip_header
        self.use_side_images = use_side_images
        self.correction_factor = correction_factor
        self.train_valid_split = train_valid_split
        self.do_flip = do_flip
        self.images = None
        self.labels = None

    def get_image_shape(self):
        img = cv2.imread(self.images[0])
        return img.shape

    def read_csv(self, labels):
        """
        Returns the lines from a driving log with base directory `dataPath`.
        If the file include headers, pass `skipHeader=True`.
        """
        lines = []
        with open(labels) as csvFile:
            reader = csv.reader(csvFile)
            if self.skip_header:
                next(reader, None)
            for line in reader:
                lines.append(line)
        return lines

    def download_data(self):
        subprocess.call(['sh', './getData.sh'])

    def extract_images(self):
        """
        Finds all the images needed for training on the path `dataPath`.
        Returns `([centerPaths], [leftPath], [rightPath], [measurement])`
        """
        if os.path.isdir(self.data_path):
            print("Training data exist")
        else:
            print("Downloading Data")
        data_directories = next(os.walk(self.data_path))[1]
        all_car_images = []
        all_steering_angles = []
        for x in data_directories:
            steering_angles = []
            car_images = []
            directory = os.path.join(self.data_path, x)
            labels_path = os.path.join(directory, self.labels_file)
            lines = self.read_csv(labels_path)
            for line in lines:
                steering_center = float(line[3])
                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + self.correction_factor
                steering_right = steering_center - self.correction_factor

                # read in images from center, left and right cameras
                img_center = os.path.join(directory, "IMG", line[0].split("/")[-1])
                img_left = os.path.join(directory, "IMG", line[1].split("/")[-1])
                img_right = os.path.join(directory, "IMG", line[2].split("/")[-1])

                # add images and angles to data set
                car_images.append(img_center)
                steering_angles.append(steering_center)
                if self.use_side_images:
                    car_images.append(img_left)
                    steering_angles.append(steering_left)
                    car_images.append(img_right)
                    steering_angles.append(steering_right)
            all_car_images.extend(car_images)
            all_steering_angles.extend(steering_angles)
        print('Total Images: {}'.format(len(all_car_images)))
        return all_car_images, all_steering_angles

    def generator(self, samples, batch_size=32):
        """
        Generate the required images and measurements for training/
        `samples` is a list of pairs (`imagePath`, `measurement`).
        """
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            samples = sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []
                for imagePath, measurement in batch_samples:
                    original_image = cv2.imread(imagePath)
                    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    angles.append(measurement)

                    if self.do_flip:
                        # Flipping
                        images.append(cv2.flip(image, 1))
                        angles.append(measurement * -1.0)

                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    def splitData(self):
        self.images, self.labels = self.extract_images()
        samples = list(zip(self.images, self.labels))
        train_samples, validation_samples = train_test_split(samples, test_size=self.train_valid_split)
        print('Train Samples: {}'.format(len(train_samples)))
        print('Validation Samples: {}'.format(len(validation_samples)))
        return train_samples, validation_samples
