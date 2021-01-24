import sys
import yaml
import os
from preprocess import PreprocessData
from model import Model
from visualization import Visualization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    if len(sys.argv) < 2:
        print("Behavioral Cloning\n", "Usage: python3 main.py config.json")
    else:
        config_file = sys.argv[1]
        with open(config_file) as yaml_file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            configs = yaml.load(yaml_file, Loader=yaml.FullLoader)

            # Data configurations
            data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), configs["data_path"])
            labels_file = configs["labels_file"]
            skip_header = bool(configs["skipHeader"])
            use_side_images = bool(configs["useSideImages"])
            correction_factor = float(configs["correctionFactor"])
            train_valid_split = float(configs["trainValidSplit"])
            do_flip = bool(configs["doFlip"])

            # Training configurations
            top_crop = int(configs["topCrop"])
            bottom_crop = int(configs["bottomCrop"])
            batch_size = int(configs["batchSize"])
            epochs = int(configs["epochs"])
            loss = configs["loss"]
            optimizer = configs["optimizer"]
            verbose = int(configs["verbose"])
            model_name = configs["modelName"]
            output_dir = configs["outputDir"]

            # Init Preprocessing
            preprocess = PreprocessData(data_path, labels_file, correction_factor, skip_header, use_side_images,
                                        train_valid_split, do_flip)

            # Preprocess data and extract training and validation samples
            train_samples, validation_samples = preprocess.splitData()

            # Initialize train and validation generators
            train_generator = preprocess.generator(train_samples, batch_size=batch_size)
            validation_generator = preprocess.generator(validation_samples, batch_size=batch_size)

            # Get image shape
            img_shape = preprocess.get_image_shape()

            # Initialize training network
            network = Model(img_shape, top_crop, bottom_crop, batch_size, epochs, loss, optimizer, verbose,
                            train_generator,
                            validation_generator, train_samples, validation_samples, model_name)

            model = network.create_model()

            # Initialize visualization
            visualize = Visualization(model, output_dir)
            visualize.visualize_model()

            network.get_summary(model)

            results = network.train_model(model)

            visualize.save_plots(results)




if __name__ == "__main__":
    main()
