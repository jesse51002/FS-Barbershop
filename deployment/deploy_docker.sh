export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1


cp ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth vgg16-397923af.pth
cp ~/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth resnet18-5c106cde.pth

sm-docker build -f ./deployment/DockerfileDeploy . --repository fsbarbershop:1.0

rm -rf resnet18-5c106cde.pth
rm -rf vgg16-397923af.pth