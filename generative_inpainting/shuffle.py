import os
import random
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--training_folder_path', default='./Dataset/Training/Origin', type=str,
                    help='The training folder path')
parser.add_argument('--validation_folder_path', default='./Dataset/Validation/Origin', type=str,
                    help='The training folder path')
parser.add_argument('--train_filename', default='./data_flist/train_shuffled.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data_flist/validation_static_view.flist', type=str,
                    help='The validation filename.')


def _get_filenames(dataset_dir):
    photo_filenames = []
    image_list = os.listdir(dataset_dir)
    photo_filenames = [os.path.join(dataset_dir, _) for _ in image_list]
    return photo_filenames


if __name__ == "__main__":

    args = parser.parse_args()

    training_data_dir = args.training_folder_path
    validation_data_dir = args.validation_folder_path

    # get all file names for training set
    training_photo_filenames = _get_filenames(training_data_dir)
    print("size of training is %d" % (len(training_photo_filenames)))
    # get all file names for validation set
    validation_photo_filenames = _get_filenames(validation_data_dir)
    print("size of validation is %d" % (len(validation_photo_filenames)))

    # shuffle
    random.seed(0)
    random.shuffle(training_photo_filenames)
    random.shuffle(validation_photo_filenames)

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    if not os.path.exists(args.validation_filename):
        os.mknod(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_photo_filenames))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_photo_filenames))
    fo.close()
