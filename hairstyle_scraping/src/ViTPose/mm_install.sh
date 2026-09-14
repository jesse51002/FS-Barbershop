cd ./src/
git clone https://github.com/open-mmlab/mmcv.git
git checkout v1.3.9
cd mmcv
MMCV_WITH_OPS=1 pip install -e .
cd ../../
# rm -rf mmcv

cd ./src/
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install --no-cache-dir -e .
cd ../../
# rm -rf mmpose

pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python-headless