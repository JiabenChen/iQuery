# System libs
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7" 
import random
import time

# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wavfile
import cv2
from mir_eval.separation import bss_eval_sources

# Our libs
from arguments import ArgParser
from dataset import MUSICMixNomotionDataset
from models import ModelBuilder, activate
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap, \
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs
from viz import plot_loss_metrics, HTMLVisualizer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, m):

        return torch.mean(1.0/(1e-6 + torch.abs(m-0.5)))

def tf_data(x, B, s1, s2):
    return torch.softmax(x, dim=-1)[:, 0].view(B, 1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, s1, s2)

# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, maskformer, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame = nets
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1)
        self.crit = crit
        self.cts = nn.CrossEntropyLoss()
        self.bn = LBSign.apply
        self.cts_bn = BinaryLoss()
        self.net_maskformer = maskformer
       

    def forward(self, batch_data, args, cur_num_mix=2):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags'] 
        frames = batch_data['frames'] 
        classes = batch_data['classes'] 
        mag_mix = mag_mix + 1e-10

        N = cur_num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # 0.2 ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)

        # LOG magnitude
        log_mag_mix = torch.log1p(mag_mix).detach()

        feat_sound, feat_latent = self.net_sound(log_mag_mix)
        feat_sound = activate(feat_sound, args.sound_activation)
        feat_latent = activate(feat_latent, args.sound_activation)

        feat_frames = [None for n in range(N)] 
        for n in range(N):
            feat_frames[n] = self.net_frame.forward_multiframe(frames[n])
            feat_frames[n] = activate(feat_frames[n], args.img_activation)

        pred_masks = [None for n in range(N)]
       
        for n in range(N):
            pred_masks[n] = self.net_maskformer(feat_latent, feat_sound, classes[n], feat_frames[n])
            pred_masks[n] = activate(pred_masks[n], args.output_activation)

        loss_sep = self.crit(pred_masks[0], gt_masks[0], weight).reshape(1)
        for i in range(1,N):
            loss_sep += self.crit(pred_masks[i], gt_masks[i], weight).reshape(1)
        loss_sep = 0.5 * loss_sep

        err = loss_sep

        return err, loss_sep, \
               {'pred_masks': pred_masks, 'gt_masks': gt_masks,
                'mag_mix': mag_mix, 'mags': mags, 'weight': weight}


# Calculate metrics
def calc_metrics(batch_data, outputs, args, cur_num_mix):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']
    pred_masks_ = outputs['pred_masks']

    # unwarp log scale
    N = cur_num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]


# Visualize predictions
def output_visuals(vis_rows, batch_data, outputs, args, cur_num_mix):
    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    frames = batch_data['frames']
    infos = batch_data['infos']

    pred_masks_ = outputs['pred_masks']
    gt_masks_ = outputs['gt_masks']
    mag_mix_ = outputs['mag_mix']
    weight_ = outputs['weight']

    # unwarp log scale
    N = cur_num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    gt_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, gt_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
            gt_masks_linear[n] = F.grid_sample(gt_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]
            gt_masks_linear[n] = gt_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    for n in range(N):
        pred_masks_[n] = pred_masks_[n].detach().cpu().numpy()
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        gt_masks_[n] = gt_masks_[n].detach().cpu().numpy()
        gt_masks_linear[n] = gt_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_[n] = (pred_masks_[n] > args.mask_thres).astype(np.float32)
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        row_elements = []

        # video names
        prefix = []
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(args.vis, prefix))

        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        mix_amp = magnitude2heatmap(mag_mix_[j, 0])
        weight = magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        cv2.imwrite(os.path.join(args.vis, filename_mixmag), mix_amp[::-1, :, :])
        cv2.imwrite(os.path.join(args.vis, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(args.vis, filename_mixwav), args.audRate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):

            # GT and predicted audio recovery
            gt_mag = mag_mix[j, 0] * gt_masks_linear[n][j, 0]
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

            # output masks
            filename_gtmask = os.path.join(prefix, 'gtmask{}.jpg'.format(n+1))
            filename_predmask = os.path.join(prefix, 'predmask{}.jpg'.format(n+1))
            gt_mask = (np.clip(gt_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask = (np.clip(pred_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.vis, filename_gtmask), gt_mask[::-1, :])
            cv2.imwrite(os.path.join(args.vis, filename_predmask), pred_mask[::-1, :])

            # ouput spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n+1))
            filename_predmag = os.path.join(prefix, 'predamp{}.jpg'.format(n+1))
            gt_mag = magnitude2heatmap(gt_mag)
            pred_mag = magnitude2heatmap(pred_mag)
            cv2.imwrite(os.path.join(args.vis, filename_gtmag), gt_mag[::-1, :, :])
            cv2.imwrite(os.path.join(args.vis, filename_predmag), pred_mag[::-1, :, :])

            # output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n+1))
            filename_predwav = os.path.join(prefix, 'pred{}.wav'.format(n+1))
            wavfile.write(os.path.join(args.vis, filename_gtwav), args.audRate, gt_wav)
            wavfile.write(os.path.join(args.vis, filename_predwav), args.audRate, preds_wav[n])

            filename_av = os.path.join(prefix, 'av{}.mp4'.format(n+1))

            row_elements += [
                {'video': filename_av},
                {'image': filename_predmag, 'audio': filename_predwav},
                {'image': filename_gtmag, 'audio': filename_gtwav},
                {'image': filename_predmask},
                {'image': filename_gtmask}]

        row_elements += [{'image': filename_weight}]
        vis_rows.append(row_elements)


def evaluate(netWrapper, loader, history, epoch, args, cur_num_mix):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=True)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # initialize HTML header
    visualizer = HTMLVisualizer(os.path.join(args.vis, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, cur_num_mix+1):
        header += ['Video {:d}'.format(n),
                   'Predicted Audio {:d}'.format(n),
                   'GroundTruth Audio {}'.format(n),
                   'Predicted Mask {}'.format(n),
                   'GroundTruth Mask {}'.format(n)]
    header += ['Loss weighting']
    visualizer.add_header(header)
    vis_rows = []

    for i, batch_data in enumerate(loader):
        # forward pass
        err,_, outputs = netWrapper.forward(batch_data, args, cur_num_mix)
        err = err.mean()

        loss_meter.update(err.item())
        print('[Eval] iter {}, loss: {:.4f}'.format(i, err.item()))

        sdr_mix, sdr, sir, sar = calc_metrics(batch_data, outputs, args, cur_num_mix)

        sdr_mix_meter.update(sdr_mix)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)

        # output visualization
        output_visuals(vis_rows, batch_data, outputs, args, cur_num_mix)


    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, '
          'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
          .format(epoch, loss_meter.average(),
                  sdr_mix_meter.average(),
                  sdr_meter.average(),
                  sir_meter.average(),
                  sar_meter.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())

    print('Plotting html for visualization...')
    visualizer.add_rows(vis_rows)
    visualizer.write_html()

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history)


# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, mask_optimizer, mask_lr_scheduler, args, cur_num_mix):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        # forward pass
        netWrapper.zero_grad()
        err,loss,_= netWrapper.forward(batch_data, args, cur_num_mix)
        err = err.mean()

        # backward
        err.backward()
        optimizer.step()
        mask_optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, lr_maskformer: {}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, mask_optimizer.state_dict()['param_groups'][0]['lr'],
                          err.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(loss.mean().item())
    mask_lr_scheduler.step()


def checkpoint(nets, maskformer, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_frame) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(args.ckpt, suffix_latest))
    torch.save(maskformer.state_dict(),
               '{}/maskformer_{}'.format(args.ckpt, suffix_latest))

    cur_err = history['val']['err'][-1]
    cur_sdr = history['val']['sdr'][-1]
    if cur_sdr > args.best_sdr:
        args.best_sdr = cur_sdr

        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_best))
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_best))
        torch.save(maskformer.state_dict(),
               '{}/maskformer_{}'.format(args.ckpt, suffix_best))


def create_optimizer(nets, args):
    (net_sound, net_frame) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_frame.fc.parameters(), 'lr': args.lr_sound}]
    return torch.optim.Adam(param_groups)

def maskformer_optimizer(model, args):
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]}
    ]
    return torch.optim.AdamW(param_dicts, lr=args.lr_maskformer, weight_decay=args.weight_decay_maskformer)

def adjust_maskformer_learning_rate(optimizer, args):
    return torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_maskformer)

def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound)
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        weights=args.weights_frame)

    nets = (net_sound, net_frame)
    maskformer = builder.build_maskformer(
        in_channels=args.in_channels,
        hidden_dim=args.MASK_FORMER_HIDDEN_DIM,
        num_queries=args.MASK_FORMER_NUM_OBJECT_QUERIES,
        nheads=args.MASK_FORMER_NHEADS,
        dropout=args.MASK_FORMER_DROPOUT,
        dim_feedforward=args.MASK_FORMER_DIM_FEEDFORWARD,
        enc_layers=args.MASK_FORMER_ENC_LAYERS,
        dec_layers=args.MASK_FORMER_DEC_LAYERS,
        mask_dim=args.SEM_SEG_HEAD_MASK_DIM,
        weights = args.weights_maskformer
    )
    crit = builder.build_criterion(arch=args.loss)

    # Dataset and Loader
    dataset_train = MUSICMixNomotionDataset(
        args.list_train, args.num_mix, args, split='train')
    dataset_val = MUSICMixNomotionDataset(
        args.list_val, args.num_mix, args, max_sample=args.num_val, split=args.split)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # Wrap networks
    netWrapper = NetWrapper(maskformer, nets, crit)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)


    # Set up optimizer
    optimizer = create_optimizer(nets, args)
    mask_optimizer = maskformer_optimizer(maskformer, args)
    mask_lr_scheduler = adjust_maskformer_learning_rate(mask_optimizer,args)
    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}}


    # Eval mode
    if args.mode == 'eval':
        args.testing = True
        evaluate(netWrapper, loader_val, history, 0, args)
        print('Evaluation Done!')
        return

    # Training loop
    for epoch in range(1, args.num_epoch + 1):
        cur_num_mix = args.num_mix
        train(netWrapper, loader_train, optimizer, history, epoch, mask_optimizer, mask_lr_scheduler, args, cur_num_mix)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            args.testing = True
            evaluate(netWrapper, loader_val, history, epoch, args, args.num_mix)
            args.testing = False
            # checkpointing
            checkpoint(nets, maskformer, history, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    # experiment name
    if args.mode == 'train':
        args.id += '-ExampleMUSICNomotion'
        args.id += '-{}mix'.format(args.num_mix)
        args.id += '-{}decoder'.format(args.MASK_FORMER_DEC_LAYERS)
        args.id += '-{}inchannel'.format(args.in_channels)
        args.id += '-{}numquery'.format(args.MASK_FORMER_NUM_OBJECT_QUERIES)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}'.format(
            args.arch_frame, args.arch_sound)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        if args.binary_mask:
            assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
            args.id += '-binary'
        else:
            args.id += '-ratio'
        if args.weighted_loss:
            args.id += '-weightedLoss'
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)
    elif args.mode == 'eval':
        args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
        args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')
        args.weights_maskformer = os.path.join(args.ckpt, 'maskformer_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")
    args.best_sdr = float("-inf")
    args.testing = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
