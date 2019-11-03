from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import pprint
import time

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
import cv2
from frcnn.model import test
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sacred import Experiment
from tracktor.config import get_output_dir, cfg
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.resnet import resnet50
from tracktor.tracker import Tracker
from tracktor.utils import interpolate, plot_sequence, plot_tracks
from tracktor.transform_model import my_transform

# Torch 0.3.1 to load model of Torch 0.4.0
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')


# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_network_config'])
ex.add_config(ex.configurations[0]._conf['tracktor']['obj_detect_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')

# Tracker = ex.capture(Tracker, prefix='tracker.tracker')


@ex.automain
def my_main(tracktor, siamese, _config):
    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection

    print("[*] Building object detector")
    print("tracktor['network'] is: ", tracktor['network'])

    if tracktor['network'].startswith('frcnn'):
        # FRCNN
        from tracktor.frcnn import FRCNN
        from frcnn.model import config

        if _config['frcnn']['cfg_file']:
            config.cfg_from_file(_config['frcnn']['cfg_file'])
        if _config['frcnn']['set_cfgs']:
            config.cfg_from_list(_config['frcnn']['set_cfgs'])

        obj_detect = FRCNN(num_layers=101)
        obj_detect.create_architecture(2, tag='default',
            anchor_scales=config.cfg.ANCHOR_SCALES,
            anchor_ratios=config.cfg.ANCHOR_RATIOS)
        state_dict_person = torch.load(tracktor['obj_detect_weights_person'])
        obj_detect.load_state_dict(state_dict_person)
        # loading head-detection model
        obj_detect_head = FRCNN(num_layers=101)
        obj_detect_head.create_architecture(2, tag='default',
            anchor_scales=config.cfg.ANCHOR_SCALES,
            anchor_ratios=config.cfg.ANCHOR_RATIOS)
        state_dict_head = torch.load(tracktor['obj_detect_weights_head'])
        state_dict_head = my_transform(state_dict_head)
        obj_detect_head.load_state_dict(state_dict_head)


    elif tracktor['network'].startswith('mask-rcnn'):
        # MASK-RCNN
        pass

    elif tracktor['network'].startswith('fpn'):
        # FPN
        from tracktor.fpn import FPN
        from fpn.model.utils import config
        config.cfg.TRAIN.USE_FLIPPED = False
        config.cfg.CUDA = True
        config.cfg.TRAIN.USE_FLIPPED = False
        checkpoint = torch.load(tracktor['obj_detect_weights'])

        if 'pooling_mode' in checkpoint.keys():
            config.cfg.POOLING_MODE = checkpoint['pooling_mode']

        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                    'ANCHOR_RATIOS', '[0.5,1,2]']
        config.cfg_from_file(_config['tracktor']['obj_detect_config'])
        config.cfg_from_list(set_cfgs)

        obj_detect = FPN(('__background__', 'pedestrian'), 101, pretrained=False)
        obj_detect.create_architecture()

        obj_detect.load_state_dict(checkpoint['model'])
    else:
        raise NotImplementedError(f"Object detector type not known: {tracktor['network']}")

    pprint.pprint(config.cfg)
    obj_detect.eval()
    obj_detect.cuda()
    obj_detect_head.eval()
    obj_detect_head.cuda()

    # reid
    reid_network = resnet50(pretrained=False, **siamese['cnn'])
    reid_network.load_state_dict(torch.load(tracktor['reid_network_weights']))
    reid_network.eval()
    reid_network.cuda()

    # tracktor
    if 'oracle' in tracktor:
        tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        print(tracktor['tracker'])
        tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

    print("[*] Beginning evaluation...")

    time_total = 0
    tracker.reset()
    now = time.time()

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("test", 800, 600)

    seq_name = 'MOT-2'
    video_file = osp.join(cfg.ROOT_DIR, 'video/' + seq_name + '.mp4')
    print("[*] Evaluating: {}".format(video_file))

    # ===============================================
    # transform each video frame to main frame format
    # ===============================================
    transforms = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    vdo = cv2.VideoCapture()
    vdo.open(video_file)
    im_width = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    area = 0, 0, im_width, im_height

    print("===video frame's area:", area)
    # video = cv2.VideoCapture(video_file)
    # if not video.isOpened():
    #     print("error opening video stream or file!")

    # while (video.isOpened()):
    while vdo.grab():
        _, frame = vdo.retrieve()

        # success, frame = video.read()
        # if not success:
        #     break
        # print(frame)  # (540, 960, 3)

        blobs, im_scales = test._get_blobs(frame)
        data = blobs['data']

        # print(data.shape)  # (1, 562, 1000, 3)
        # print(im_scales)  # [1.04166667]

        sample = {}
        sample['image'] = cv2.resize(frame, (0, 0), fx=im_scales, fy=im_scales, interpolation=cv2.INTER_NEAREST)
        sample['im_path'] = video_file
        sample['data'] = torch.from_numpy(data).unsqueeze(0)
        im_info = np.array([data.shape[1], data.shape[2], im_scales[0]], dtype=np.float32)
        sample['im_info'] = torch.from_numpy(im_info).unsqueeze(0)

        # convert to siamese input
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transforms(frame)
        # print(frame.shape)  # torch.Size([3, 540, 960])

        sample['app_data'] = frame.unsqueeze(0).unsqueeze(0)
        # print(sample['app_data'].size())  # torch.Size([1, 1, 3, 540, 960])

        # additional info
        # sample['gt'] = {}
        # sample['vis'] = {}
        # sample['dets'] = []

        # print('frame begin')
        # print(sample)
        # print('frame end')

        tracker.step(sample)
        tracker.show_tracks(area)

    video.release()
    print('the current video' + video_file + ' is done')

    results = tracker.get_results()
    time_total += time.time() - now
    print("[*] Tracks found: {}".format(len(results)))
    print("[*] Time needed for {} evaluation: {:.3f} s".format(seq_name, time.time() - now))

    # print('this is : ' + tracktor['dataset'])

    # for sequence in Datasets(tracktor['dataset']):
    # #for sequence in Datasets('MOT-02'):
    #
    #     print('sequence---------', type(sequence), len(sequence))
    #
    #     tracker.reset()
    #     now = time.time()
    #
    #     print("[*] Evaluating: {}".format(sequence))
    #
    #     data_loader = DataLoader(sequence, batch_size=1, shuffle=False)
    #     for i, frame in enumerate(data_loader):
    #
    #         print('frame begin')
    #         print(frame)
    #         print('frame end')
    #
    #         if i >= len(sequence) * tracktor['frame_split'][0] and i <= len(sequence) * tracktor['frame_split'][1]:
    #
    #             tracker.step(frame)
    #     results = tracker.get_results()
    #
    #
    #     time_total += time.time() - now
    #
    #     print("[*] Tracks found: {}".format(len(results)))
    #     print("[*] Time needed for {} evaluation: {:.3f} s".format(sequence, time.time() - now))
    #
    #     if tracktor['interpolate']:
    #         results = interpolate(results)
    #
    #     plot_tracks(sequence, results)
    #     sequence.write_results(results, osp.join(output_dir))
    #
    #     if tracktor['write_images']:
    #        plot_sequence(results, sequence, osp.join(output_dir, tracktor['dataset'], str(sequence)))
    print("[*] Evaluation for all sets (without image generation): {:.3f} s".format(time_total))
