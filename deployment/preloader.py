# fused relu and upfirdn2d ninja build takes a while to preload them in order to avoid timeout in model creation
import sys
sys.path.insert(0, '/opt/ml/code')
import os

# Removes lock file as they can result in indefinite freezing
# https://github.com/zhou13/neurvps/issues/1
fused_lock_pth = '/root/.cache/torch_extensions/py310_cu118/fused/lock'
upfirdn2d_lock_pth = '/root/.cache/torch_extensions/py310_cu118/upfirdn2d/lock'

if os.path.isfile(fused_lock_pth):
    print("Removing fuse lock")
    os.remove(fused_lock_pth)
    
if os.path.isfile(upfirdn2d_lock_pth):
    print("Removing upfirdn2d lock")
    os.remove(upfirdn2d_lock_pth)

# preimport net to let ninja build in the docker container before being deployed
import models.Net


import facer
"""
Preloads models to download the weights if they hadn't already been downloaded
"""
    
facer.face_parser('farl/lapa/448', device="cpu")
facer.face_detector('retinaface/mobilenet', device="cpu")
facer.face_aligner('farl/ibug300w/448', device="cpu")