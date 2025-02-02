import os, argparse, sys
import numpy as np
#%tensorflow_version 1.x
from keras.models import load_model, Model
from keras.preprocessing.image import img_to_array
from sklearn.metrics import roc_auc_score

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dm_image import DMImageDataGenerator
from dm_keras_ext import (
    load_dat_ram,
    do_2stage_training,
    DMFlush,
    DMAucModelCheckpoint
)
from dm_resnet import add_top_layers, bottleneck_org
from dm_multi_gpu import make_parallel
import warnings
#import exceptions
import builtins as exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)

print ("\n==== Predicting ====")
		
#def run(test_dir,img_size=[1152, 896], img_scale=None, rescale_factor=None,
 #       featurewise_center=True, featurewise_mean=52.16, 
  #      equalize_hist=False, augmentation=True,
   #     class_list=['neg', 'pos'], patch_net='resnet50',batch_size=64,
    #    best_model='./modelState/image_clf.h5'):
def run():
    #test_dir= "/content/gdrive/MyDrive/polygence/datasets/inbreast/validation/"
    #test_dir= "/content/gdrive/MyDrive/polygence/datasets/inbreast/dicom/"
    
    test_dir= "/content/gdrive/MyDrive/polygence/datasets/inbreast/AllDICOMs/preprocessed_images/"
    #test_dir="/content/gdrive/MyDrive/polygence/datasets/cropped/preprocessed_images/temp/train/"
    #test_dir="/content/gdrive/MyDrive/polygence/datasets/mias1/train/"
    #test_dir="/content/gdrive/MyDrive/polygence/datasets/KAUMDS/processed/"
    img_size=[1152, 896]
    #img_size=[3328, 4084]

    img_scale=None
    rescale_factor=0.003891
    featurewise_center=True
    featurewise_mean=44.33
    equalize_hist=False
    augmentation=True
    class_list=['neg', 'pos']
    patch_net='resnet50'
    batch_size=4
    #best_model="/content/gdrive/MyDrive/polygence/transferred_inbreast_best_model.h5"
    best_model="/content/gdrive/MyDrive/polygence/pretrained_models/inbreast_vgg16_[512-512-1024]x2_hybrid.h5"
    #best_model="/content/gdrive/MyDrive/polygence/pretrained_models/ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5"
    '''Train a deep learning model for image classifications
    '''

    # ======= Environmental variables ======== #
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    nb_worker = int(os.getenv('NUM_CPU_CORES', 4))
    gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))

    # ========= Image generator ============== #
    if featurewise_center:
        test_imgen = DMImageDataGenerator(featurewise_center=True)
        test_imgen.mean = featurewise_mean
    else:
        test_imgen = DMImageDataGenerator()
		
    if patch_net != 'yaroslav':
        dup_3_channels = True
    else:
        dup_3_channels = False
   
    # ==== Predict on test set ==== #
    print ("\n==== Predicting on test set ====")
    test_imgen.mean = 52.18
    test_generator = test_imgen.flow_from_directory(
        test_dir, target_size=img_size, target_scale=img_scale,
        rescale_factor=rescale_factor,
        equalize_hist=equalize_hist, dup_3_channels=True, 
        classes=class_list, class_mode='categorical', batch_size=batch_size, 
        shuffle=False)
    test_samples = test_generator.nb_sample
    #### DEBUG ####
    # test_samples = 5
    #### DEBUG ####
    print ("Test samples =", test_samples)
    print ("Load saved best model:", best_model + '.')
    sys.stdout.flush()
	
	# Create a basic model instance
    #model = create_model()
    #resume_from="/content/gdrive/MyDrive/polygence/pretrained_models/ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5"

    #model = load_model(resume_from, compile=False)

    model = load_model(best_model,compile=False)
    
	#image = img_to_array(test_generator.next)
    #image = np.expand_dims(image, axis=0)
    #pred = model.predict(image)

    print ("Done. ")
    # test_steps = int(test_generator.nb_sample/batch_size)
    # test_res = image_model.evaluate_generator(
    #     test_generator, test_steps, nb_worker=nb_worker, 
    #     pickle_safe=True if nb_worker > 1 else False)
    #test_auc = DMAucModelCheckpoint.calc_test_auc(
        #test_generator, model, test_samples=test_samples,return_y_res=True)
    res_auc, res_y_true, res_y_pred = DMAucModelCheckpoint.calc_test_auc(
        test_generator, model, test_samples=test_samples,return_y_res=True,test_augment=True)
        
    print ("Ritvik: AUROC on test set with AUGMENT:", res_auc)
    
    #all_mod_y_pred_avg = (res_y_pred[:,1] + vgg_y_pred[:,1] + hybrid_y_pred[:,1])/3
    #print roc_auc_score(y_true:res_y_true[:,1], y_score:res_y_pred)

    
	
	
if __name__ == '__main__':
    print ("\n>>>  testing ....: <<<\n")
    run()
	
	
	
