import argparse
import time
import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='generative_inpainting/Dataset/Testing/Origin', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='generative_inpainting/Dataset/Testing/Mask', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='generative_inpainting/Dataset/Testing/Results', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='generative_inpainting/logs/full_model10', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    #ng.get_gpus(1)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    FLAGS = ng.Config('generative_inpainting/inpaint.yml')
    # os.environ['CUDA_VISIBLE_DEVICES'] =''
    args = parser.parse_args()
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    #tf.get_variable_scope().reuse_variables()
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, 128, 128*2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')
    
    t = time.time()
    filenames = glob.glob(args.image + '/*.jpg')
    for i in range(len(filenames)):
        image = cv2.imread(filenames[i])
        mask = cv2.imread(args.mask + '/' + filenames[i].split('/')[-1])
        #print(args.mask + '/' + filenames[i].split('/')[-1])
        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 4
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        #print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        #print('Processed: {}'.format(./Dataset/Testing/Results/'+ filenames[i].split('/')[4]))
        cv2.imwrite(args.output + '/' + filenames[i].split('/')[-1], result[0][:, :, ::-1])

    print('Time total: {}'.format(time.time() - t))