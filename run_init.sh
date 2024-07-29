source ~/.bashrc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
conda remove --name OWDet --all	

python setup.py build develop

pkg-resources==0.0.0
dataclasses==0.8
cd maskcut
python demo.py --img-path imgs/demo2.jpg \
  --N 3 --tau 0.15 --vit-arch base --patch-size 8 \
  [--other-options]
pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
-i https://mirrors.bfsu.edu.cn/pypi/web/simple
---------------------------Order-------------------------------------

python tools/train_net.py --num-gpus 2 --eval-only --config-file ./configs/OWDet/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "./output/t1_final/model_final.pth"
python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWDet/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "./output/t1_final/model_final.pth"

python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWDet/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "./output/t1_final/model_final.pth"


python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52125' --resume --config-file ./configs/OWDet/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output_mr/t1"

python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWDet/t1/t1_val.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWDet.TEMPERATURE 1.5 OUTPUT_DIR "./output_mr/t1_final" MODEL.WEIGHTS "./output_mr/t1/model_final.pth"


---------------------OWDet-ENV----------------------------------
conda create -n OWDet  python=3.7.0 -y 
conda activate OWDet
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

python -m pip install -e OWDet -i https://mirrors.bfsu.edu.cn/pypi/web/simple


pip需要的其他安装的库按照requirement中的版本安装:
pip install reliability==0.5.6 -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install shortuuid==1.0.1 -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install reliability==0.5.6 -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install reliability==0.8.12 -i https://mirrors.bfsu.edu.cn/pypi/web/simple
python 3.8.17
------------------OWDet-PROBLEM---------------------------
AttributeError: module ‘PIL.Image‘ has no attribute ‘LINEAR‘:
在.....detectron2/data/transforms/transform.py 脚本的第46行的LINEAR替换为BILINEAR即可


AttributeError: module 'numpy' has no attribute 'str':
把np.str改成np.str_


raise ValueError('The value argument must be within the support')
ValueError: The value argument must be within the support:
在某个函数或方法中，传递的值超出了其支持的范围


RuntimeError: radix_sort: failed on 1st step: cudaErrorInvalidDevice: invalid device ordinal:
问题：错误代码为"cudaErrorInvalidDevice"，表示设备顺序无效。
解决方法：
去掉device=negative.device，变为：
perm1 = torch.randperm(positive_numel, device=positive.device)[:num_pos]
perm2 = torch.randperm(negative_numel)[:num_neg] 
perm2 = perm2.to(negative.device)












