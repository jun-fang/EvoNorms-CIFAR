# # CUDA_VISIBLE_DEVICES=0 python3 main.py -b=256

python3 main.py -a=resnet18 -nal=BN-ReLU -b=256
python3 main.py -a=resnet18 -nal=BN-ReLU -b=128
python3 main.py -a=resnet18 -nal=BN-ReLU -b=64
python3 main.py -a=resnet18 -nal=BN-ReLU -b=32
python3 main.py -a=resnet18 -nal=BN-ReLU -b=16
python3 main.py -a=resnet18 -nal=BN-ReLU -b=8
python3 main.py -a=resnet18 -nal=BN-ReLU -b=4

python3 main.py -a=resnet18 -nal=GN-ReLU -b=256
python3 main.py -a=resnet18 -nal=GN-ReLU -b=128
python3 main.py -a=resnet18 -nal=GN-ReLU -b=64
python3 main.py -a=resnet18 -nal=GN-ReLU -b=32
python3 main.py -a=resnet18 -nal=GN-ReLU -b=16
python3 main.py -a=resnet18 -nal=GN-ReLU -b=8
python3 main.py -a=resnet18 -nal=GN-ReLU -b=4

python3 main.py -a=resnet18 -nal=EvoNormS0 -b=256
python3 main.py -a=resnet18 -nal=EvoNormS0 -b=128
python3 main.py -a=resnet18 -nal=EvoNormS0 -b=64
python3 main.py -a=resnet18 -nal=EvoNormS0 -b=32
python3 main.py -a=resnet18 -nal=EvoNormS0 -b=16
python3 main.py -a=resnet18 -nal=EvoNormS0 -b=8
python3 main.py -a=resnet18 -nal=EvoNormS0 -b=4

python3 main.py -a=resnet18 -nal=EvoNormB0 -b=256
python3 main.py -a=resnet18 -nal=EvoNormB0 -b=128
python3 main.py -a=resnet18 -nal=EvoNormB0 -b=64
python3 main.py -a=resnet18 -nal=EvoNormB0 -b=32
python3 main.py -a=resnet18 -nal=EvoNormB0 -b=16
python3 main.py -a=resnet18 -nal=EvoNormB0 -b=8
python3 main.py -a=resnet18 -nal=EvoNormB0 -b=4