import argparse


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Model related arguments
        parser.add_argument('--id', default='',
                            help="a name for identifying the model")
        parser.add_argument('--num_mix', default=2, type=int,
                            help="number of sounds to mix")
        parser.add_argument('--arch_sound_ground', default='vggish',
                            help="architecture of net_sound_ground")
        parser.add_argument('--arch_frame_ground', default='resnet18',
                            help="architecture of net_frame_ground")
        parser.add_argument('--arch_sound', default='unet7',
                            help="architecture of net_sound")
        parser.add_argument('--arch_frame', default='resnet18dilated',
                            help="architecture of net_frame")
        parser.add_argument('--arch_synthesizer', default='linear',
                            help="architecture of net_synthesizer")
        parser.add_argument('--arch_grounding', default='base',
                            help="architecture of net_grounding")
        parser.add_argument('--weights_sound_ground', default='',
                            help="weights to finetune net_sound_ground")
        parser.add_argument('--weights_frame_ground', default='',
                            help="weights to finetune net_frame_ground")
        parser.add_argument('--weights_sound', default='',
                            help="weights to finetune net_sound")
        parser.add_argument('--weights_frame', default='',
                            help="weights to finetune net_frame")
        parser.add_argument('--weights_synthesizer', default='',
                            help="weights to finetune net_synthesizer")
        parser.add_argument('--weights_grounding', default='',
                            help="weights to finetune net_grounding")
        parser.add_argument('--weights_maskformer', default='',
                            help="weights to finetune maskformer")
        parser.add_argument('--num_channels', default=32, type=int,
                            help='number of channels')
        # parser.add_argument('--num_frames', default=1, type=int,
        #                     help='number of frames')
        parser.add_argument('--num_frames', default=64, type=int,
                            help='number of frames') #ave
        parser.add_argument('--stride_frames', default=1, type=int,
                            help='sampling stride of frames')
        parser.add_argument('--img_pool', default='maxpool',
                            help="avg or max pool image features")
        parser.add_argument('--img_activation', default='sigmoid',
                            help="activation on the image features")
        parser.add_argument('--sound_activation', default='no',
                            help="activation on the sound features")
        parser.add_argument('--output_activation', default='sigmoid',
                            help="activation on the output")
        parser.add_argument('--binary_mask', default=0, type=int,
                            help="whether to use bianry masks")
        parser.add_argument('--mask_thres', default=0.5, type=float,
                            help="threshold in the case of binary masks")
        parser.add_argument('--loss', default='l1',
                            help="loss function to use")
        parser.add_argument('--weighted_loss', default=0, type=int,
                            help="weighted loss")
        parser.add_argument('--log_freq', default=1, type=int,
                            help="log frequency scale")
        parser.add_argument('--split', default='val',
                            help="val or test")

        # Data related arguments
        parser.add_argument('--num_gpus', default=1, type=int,
                            help='number of gpus to use')
        parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                            help='input batch size')
        parser.add_argument('--workers', default=16, type=int,
                            help='number of data loading workers')
        parser.add_argument('--num_val', default=-1, type=int,
                            help='number of images to evalutate')
        parser.add_argument('--num_vis', default=40, type=int,
                            help='number of images to evalutate')

        parser.add_argument('--audLen', default=65535, type=int,
                           help='sound length for MUSIC')
        # parser.add_argument('--audLen', default=47087, type=int,
        #                     help='sound length') #ave
        # parser.add_argument('--audRate', default=22050, type=int,
        #                     help='sound sampling rate') #ave
        parser.add_argument('--audRate', default=11025, type=int,
                            help='sound sampling rate')
        parser.add_argument('--stft_frame', default=1022, type=int,
                            help="stft frame length")
        parser.add_argument('--stft_hop', default=256, type=int,
                            help="stft hop length")
        # parser.add_argument('--stft_hop', default=184, type=int,
        #                     help="stft hop length") #ave

        parser.add_argument('--imgSize', default=224, type=int,
                            help='size of input frame')
        # parser.add_argument('--frameRate', default=8, type=float,
        #                     help='video frame sampling rate')
        parser.add_argument('--frameRate', default=25, type=float,
                            help='video frame sampling rate') #ave

        # Misc arguments
        parser.add_argument('--seed', default=1234, type=int,
                            help='manual seed')
        parser.add_argument('--ckpt', default='../data/ckpt',
                            help='folder to output checkpoints')
        parser.add_argument('--disp_iter', type=int, default=20,
                            help='frequency to display')
        parser.add_argument('--eval_epoch', type=int, default=1,
                            help='frequency to evaluate')
        
        #Maskformer arguments
        parser.add_argument('--in_channels', default=32, type=int,
                            help='channels of the input features')
        parser.add_argument('--MASK_FORMER_HIDDEN_DIM', default=32, type=int,
                            help='Transformer feature dimension')  
        parser.add_argument('--MASK_FORMER_NUM_OBJECT_QUERIES', default=20, type=int,
                            help='number of queries')
        parser.add_argument('--MASK_FORMER_NHEADS', default=8, type=int,
                            help='number of heads')  
        parser.add_argument('--MASK_FORMER_DROPOUT', default=0.1, type=float,
                            help='dropout in Transformer')  
        parser.add_argument('--MASK_FORMER_DIM_FEEDFORWARD', default=256, type=int,
                            help='feature dimension in feedforward network')  
        parser.add_argument('--MASK_FORMER_ENC_LAYERS', default=0, type=int,
                            help='number of Transformer encoder layers')  
        parser.add_argument('--MASK_FORMER_DEC_LAYERS', default=6, type=int,
                            help='number of Transformer decoder layers')  
        parser.add_argument('--SEM_SEG_HEAD_MASK_DIM', default=32, type=int,
                            help='mask feature dimension')  

        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser

        parser.add_argument('--mode', default='train',
                            help="train/eval")
        parser.add_argument('--list_train',
                            default='data/train.csv')
        parser.add_argument('--list_val',
                            default='data/val.csv')
        parser.add_argument('--dup_trainset', default=40, type=int,
                            help='duplicate so that one epoch has more iters')

        # optimization related arguments
        parser.add_argument('--num_epoch', default=100, type=int,
                            help='epochs to train for')
        parser.add_argument('--lr_frame_ground', default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_sound_ground', default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_frame', default=1e-4, type=float, help='LR')
        parser.add_argument('--lr_sound', default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_synthesizer',
                            default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_grounding',
                            default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_maskformer',
                            default=1e-4, type=float, help='LR')
        parser.add_argument('--lr_steps',
                            nargs='+', type=int, default=[40, 60],
                            help='steps to drop LR in epochs')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weights regularizer')
        parser.add_argument('--weight_decay_maskformer', default=1e-4, type=float,
                            help='weights regularizer')
        parser.add_argument('--lr_drop_maskformer', default=60, type=int,
                            help='lr_drop')
        parser.add_argument('--lr_drop_maskformer_1', default=60, type=int,
                            help='lr_drop')
        parser.add_argument('--lr_drop_maskformer_2', default=80, type=int,
                            help='lr_drop')                    
                            
        self.parser = parser

    def print_arguments(self, args):
        print("Input arguments:")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))

    def parse_train_arguments(self):
        self.add_train_arguments()
        args = self.parser.parse_args()
        self.print_arguments(args)
        return args
