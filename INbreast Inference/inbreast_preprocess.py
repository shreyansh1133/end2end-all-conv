import argparse
import os
import pickle
from functools import partial
from multiprocessing import Pool

from wand.image import Image


def preprocess(exam, data_folder, save_path, image_format):
    """Preprocess the image and save to a directory."""
    directory = '/content/gdrive/MyDrive/polygence/datasets/inbreast/AllPNGs'
    print("***data_folder  is :",data_folder)

    for filename in os.listdir(data_folder):
        print("***preprocess img is :",filename)
        im_path = os.path.join(data_folder, filename)
                    #image_path = data_folder + '/' + image + '.' + image_format
                # Extract subdirectories
        subdirs = "/".join(filename.split('/')[:-1])
        save_dirs = os.path.join(save_path, subdirs)
                # Extract image id
        image_id = filename.split('/')[-1]
                # Create save directories
        print("***image_id  is :",image_id)

        os.makedirs(save_dirs, exist_ok=True)
        png_save_path = os.path.join(save_dirs, image_id + '.png')

        with Image(filename=im_path, format=image_format) as img:
            print("***Image")
            with img.clone() as i:
                i.auto_level()
                with i.convert('png') as png_image:
                    png_image.transform(resize='896x1152!')
                    print("***preprocess , the trnasfornmed img dire is :",png_save_path)

                    png_image.save(filename=png_save_path)


def main(initial_exam_list_path, data_folder, preprocessed_folder, num_processes, preprocess_flag, image_format):

    if os.path.isdir(preprocessed_folder) and preprocess_flag == "False":
        print("The images are already preprocessed.")
    else:
        print("Preprocessing images.")
        with open(initial_exam_list_path, "rb") as f:
            d = pickle.load(f)
        # Use multiprocessing to split into chunks
        # Create partial function for pool
        f = partial(preprocess, data_folder=data_folder, save_path=preprocessed_folder, image_format=image_format)
        p = Pool(num_processes)
        p.map(f, d)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for end2end model")
    parser.add_argument('--input-data-folder', required=True)
    parser.add_argument('--exam-list-path', required=True)
    parser.add_argument('--preprocessed-folder', required=True)
    parser.add_argument('--num-processes', required=True, type=int)
    parser.add_argument('--always-preprocess', required=True)
    parser.add_argument('--image-format', required=True)

    args = parser.parse_args()
    main(args.exam_list_path, args.input_data_folder, args.preprocessed_folder, args.num_processes, args.always_preprocess, args.image_format)

