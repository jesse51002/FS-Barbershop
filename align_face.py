import dlib
from pathlib import Path
import argparse
import torchvision
from utils.drive import open_url
from utils.shape_predictor import align_face
import PIL
import os

parser = argparse.ArgumentParser(description='Align_face')

parser.add_argument('-unprocessed_dir', type=str, default='unprocessed', help='directory with unprocessed images')
parser.add_argument('-output_dir', type=str, default='input/face', help='output directory')

parser.add_argument('-output_size', type=int, default=1024, help='size to downscale the input images to, must be power of 2')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')
parser.add_argument('--overwrite', action='store_true', default=False, help='Whether to overwrite')

###############
parser.add_argument('-inter_method', type=str, default='bicubic')


accepted_format = ["jpeg", "jpg", "png"]

args = parser.parse_args()

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True,exist_ok=True)

print("Downloading Shape Predictor")
f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
print(f)
predictor = dlib.shape_predictor(f)

already_exists = [x.split(".")[0] for x in os.listdir(args.output_dir)] 

for im in Path(args.unprocessed_dir).glob("*.*"):
    str_path = str(im)
    base_name_split = os.path.basename(str_path).split(".")
    if base_name_split[-1] not in accepted_format:
        continue
    if not args.overwrite and base_name_split[0] in already_exists:
        print(f"Skipping {base_name_split[0]}, already exists")
        continue
    
    faces = align_face(str(im),predictor)

    for i,face in enumerate(faces):
        if(args.output_size):
            factor = 1024//args.output_size
            assert args.output_size*factor == 1024
            face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
            face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
            face = torchvision.transforms.ToPILImage()(face_tensor_lr)
            if factor != 1:
                face = face.resize((args.output_size, args.output_size), PIL.Image.LANCZOS)
        if len(faces) > 1:
            face.save(Path(args.output_dir) / (im.stem+f"_{i}.png"))
        else:
            face.save(Path(args.output_dir) / (im.stem + f".png"))