import os
import time
import torch
from utils.model_utils import download_weight
from threading import Thread

from args_maker import create_parser
from models.face_parsing.model import BiSeNet
from models.SegmentationMaker import SegMaker
from models.Embedding import Embedding
from models.Alignment import Alignment
from models.Blending import Blending
from models.FacerParsing import FacerModel, FacerKeypoints, FacerDetection
from models.p3m_matting.inference import human_matt_model
from models.Net import Net


def main(args):
    args.device = ["cuda:0", "cuda:0"]

    assert len(args.device) > 0, f"0 devices was supplied {args.device}"
    assert len(args.device) <= 2, f"Max of 2 devices can be supplied not {args.device}"
    
    args.is_multi_gpu = len(args.device) == 2
    
    im_path1 = os.path.join(args.input_dir, args.im_path1)
    im_path2 = os.path.join(args.input_dir, args.im_path2)
    im_path3 = os.path.join(args.input_dir, args.im_path3)
    im_set = {im_path1, im_path2, im_path3}

    print("Loading models")

    seg = BiSeNet(n_classes=16)
    seg.to(args.device[0])
    if not os.path.exists(args.seg_ckpt):
        download_weight(args.seg_ckpt)
    seg.load_state_dict(torch.load(args.seg_ckpt))
    for param in seg.parameters():
        param.requires_grad = False
    seg.eval()
    
    net0 = Net(args, device=args.device[0])
    net1 = Net(args, device=args.device[1]) if args.is_multi_gpu else None
        
    face_detector = FacerDetection()
    keypoint_model = FacerKeypoints(face_detector=face_detector, device=args.device[1] if args.is_multi_gpu else args.device[0])
    facer = FacerModel(face_detector=face_detector, device=args.device[1] if args.is_multi_gpu else args.device[0])
    background_remover = human_matt_model(device=args.device[1] if args.is_multi_gpu else args.device[0])

    ii2s = Embedding(args, net=net0)
    segmentor = SegMaker(args, facer=facer, background_remover=background_remover, keypoint_model=keypoint_model)
    align = Alignment(args, seg=seg, net0=net0, net1=net1)
    blend = Blending(args, seg=seg, net=net0, facer=facer, background_remover=background_remover)
    print("Finished loading models")

    def inverting_gpu0():
        print("Starting ai space creation")
        start = time.time()
        ii2s.invert_images_in_W([*im_set])
        ii2s.invert_images_in_FS([*im_set])
        print(f"Embedding took  {time.time() - start}")
    
    def segmentor_gpu1():
        print("Starting segmentor")
        start = time.time()
        segmentor.create_segmentations([*im_set])
        print(f"segmentor {time.time() - start}")

    if args.is_multi_gpu:
        gpu0_thread = Thread(target=inverting_gpu0)
        gpu1_thread = Thread(target=segmentor_gpu1)
    
        gpu0_thread.start()
        gpu1_thread.start()
        gpu0_thread.join()
        gpu1_thread.join()
    else:
        inverting_gpu0()
        segmentor_gpu1()
        
        
    grand_start_time = time.time()
    
    print("Starting alignment")
    start = time.time()
    align.align_images(im_path1, im_path2, sign=args.sign, align_more_region=False, smooth=args.smooth)
    if im_path2 != im_path3:
        align.align_images(im_path1, im_path3, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False)
    print(f"alignment took {time.time() - start}")

    print("Starting blending")
    start = time.time()
    blend.blend_images(im_path1, im_path2, im_path3, sign=args.sign)
    print(f"blending took {time.time() - start}")

    print(f"Total model loading time took {(time.time() - grand_start_time)} seconds")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)