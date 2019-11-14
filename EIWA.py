import os, sys
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.edge_connect import EdgeConnect

class EdgeInpaintingWithAttention():
    def __init__(self):

        self.config = self.load_config()
        # cuda visble devices
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in self.config.GPU)


        # init device
        if torch.cuda.is_available():
            print ("GPU in use...")
            self.config.DEVICE = torch.device("cuda")
            torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
        else:
            self.config.DEVICE = torch.device("cpu")

        # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
        cv2.setNumThreads(0)
        

    def main(self):

        # initialize random seed for reproduction
        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        np.random.seed(self.config.SEED)
        random.seed(self.config.SEED)


        # build the model and initialize
        model = EdgeConnect(self.config)
        model.load()

        # model training
        if self.config.MODE == 1:
            self.config.print()
            print('\nstart training...\n')
            model.train()

        # model test
        elif self.config.MODE == 2:
            self.config.print()
            print('\nstart testing...\n')
            model.test()

        # eval mode
        else:
            print('\nstart eval...\n')
            model.eval()


    def load_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', type=int, default= 1, help='1: train, 2: test, 3: eval')
        parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
        parser.add_argument('--model', type=int, choices=[1, 2, 3, 4, 5], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: inpanint fine model, 5: joint model')

        # test mode
        if parser.parse_args().mode == 2:
            parser.add_argument('--input', type=str, default='./data/ori_test/', help='path to the input images directory or an input image')
            parser.add_argument('--mask', type=str, default='./data/mask_test/', help='path to the masks directory or a mask file')
            parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
            parser.add_argument('--output', type=str, default='./checkpoints/results', help='path to the output directory')

        args = parser.parse_args()
        config_path = os.path.join(args.path, 'config.yaml')

        # create checkpoints path if does't exist
        if not os.path.exists(args.path):
            os.makedirs(args.path)

        # copy config template if does't exist
        if not os.path.exists(config_path):
            copyfile('./config.yml.example', config_path)

        # load config file
        config = Config(config_path)

        # train mode
        if args.mode == 1:
            config.MODE = 1
            if args.model:
                config.MODEL = args.model

        # test mode
        elif args.mode == 2:
            config.MODE = 2
            config.MODEL = args.model if args.model is not None else 5
            config.INPUT_SIZE = 0

            if args.input is not None:
                config.TEST_FLIST = args.input

            if args.mask is not None:
                config.TEST_MASK_FLIST = args.mask

            if args.edge is not None:
                config.TEST_EDGE_FLIST = args.edge

            if args.output is not None:
                config.RESULTS = args.output

        # eval mode
        elif args.mode == 3:
            config.MODE = 3
            config.MODEL = args.model if args.model is not None else 3

        return config


if __name__ == "__main__":
    EIWA = EdgeInpaintingWithAttention()
    EIWA.main()
