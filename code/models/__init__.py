import torch
import torchvision
import torch.nn.functional as F

from .audio_net import VGGish, ANet, UnetIquery
from .vision_net import ResnetFC, ResnetDilated, Resnet
from .criterion import BCELoss, L1Loss, L2Loss, BinaryLoss

from .maskformer_predictor import TransformerPredictor
from .maskformer_predictor_motion import TransformerPredictorMotion

def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')
    
class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_sound(self, arch='unet7', fc_dim=64, weights=''):
        # 2D models
        if arch == 'unet7':
            net_sound = UnetIquery(fc_dim=fc_dim, num_downs=7)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound

    # builder for vision
    def build_frame(self, arch='resnet18', fc_dim=64, pool_type='avgpool',
                    weights=''):
        pretrained=True
        if arch == 'resnet18fc':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetFC(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        elif arch == 'resnet18dilated':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetDilated(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net
    
    def build_maskformer(self, in_channels, hidden_dim, num_queries, nheads, dropout, dim_feedforward, enc_layers, dec_layers, mask_dim, weights=''):
        pre_norm = False
        deep_supervision = True
        enforce_input_project = False
        net = TransformerPredictor(in_channels=in_channels, hidden_dim=hidden_dim, num_queries=num_queries, nheads=nheads,dropout=dropout,dim_feedforward=dim_feedforward,enc_layers=enc_layers,dec_layers=dec_layers,pre_norm=pre_norm, mask_dim=mask_dim, deep_supervision=deep_supervision, enforce_input_project=enforce_input_project)
        if len(weights) > 0:
            print('Loading weights for maskformer')
            net.load_state_dict(torch.load(weights))
        return net
    
    def build_maskformermotion(self, in_channels, hidden_dim, num_queries, nheads, dropout, dim_feedforward, enc_layers, dec_layers, mask_dim, weights=''):
        pre_norm = False
        deep_supervision = True
        enforce_input_project = False
        net = TransformerPredictorMotion(in_channels=in_channels, hidden_dim=hidden_dim, num_queries=num_queries, nheads=nheads,dropout=dropout,dim_feedforward=dim_feedforward,enc_layers=enc_layers,dec_layers=dec_layers,pre_norm=pre_norm,mask_dim=mask_dim,deep_supervision=deep_supervision,enforce_input_project=enforce_input_project)
        if len(weights) > 0:
            print('Loading weights for maskformer')
            net.load_state_dict(torch.load(weights))
        return net
    
    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        elif arch == 'bn':
            net = BinaryLoss()
        else:
            raise Exception('Architecture undefined!')
        return net