import tensorflow as tf
import scipy.misc as misc
from networks import generator
from ContextEncoder import ContextEncoder
from utils import *
from config import *
import glob
import cv2

def main():
    if IS_TRAINED:
        #initialize the model
        input_tf = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C])
        train_phase = tf.placeholder(tf.bool)
        inpainting = generator("generator")
        patch_tf = inpainting(input_tf, train_phase)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator"))
        saver.restore(sess, "./save_para/para.ckpt")
        #saver.restore(sess, "./save_para//.\\para.ckpt")
        
        #test the model
        file_path = "./data/ori_test/*.*"
        result_path ="./baseline_test"
        filenames = glob.glob(file_path)
        for i in range(len(filenames)):
            mask, X, Y = get_mask()
            img = misc.imresize(read_img_and_crop(filenames[i]), [IMG_H, IMG_W])
            input = ((img * (1 - mask) + 255 * mask) / 127.5 - 1.0)[np.newaxis, :, :, :]
            patch = sess.run(patch_tf, feed_dict={input_tf: input, train_phase: False})
            input[0, :, :, :][X:X + MASK_H, Y:Y + MASK_W, :] = patch[0, :, :, :]
            output_image = (input[0, :, :, :]+1)*127.5
            input_image = (img * (1 - mask) + 255 * mask)
            #output = np.concatenate((img, mask*255, (input[0, :, :, :]+1)*127.5), 1)
            Image.fromarray(np.uint8(input_image)).save('/home/sunchenyu1993/CE7454/results/context_encoder_input/' + filenames[i].split('/')[-1])
            Image.fromarray(np.uint8(output_image)).save('/home/sunchenyu1993/CE7454/results/context_encoder/' + filenames[i].split('/')[-1])
            print ("saving: ",filenames[i].split('/')[-1] )
            

    else:
        CE = ContextEncoder()
        CE.train()

if __name__ == "__main__":
    main()


