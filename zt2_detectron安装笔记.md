2019.02.26记录（刘强，四川大学图像信息研究所614）
**长风破浪会有时，直挂云帆济沧海**

[安装前所需知道的知识](https://me.csdn.net/pc9803)：
a、Detectron运营商目前没有CPU实施; 需要GPU系统。
b、caffe2 已经集成到pytorch1.0中，所以框架我们直接安装pytorch即可。
c、Detectron已经过CUDA 8.0和cuDNN 6.0.21的广泛测试，不过cuda其他版本也是可以的，比如cuda9.
d、首先先保证已经安装了cuda与cudnn,若是没安装，先安装这些。
e、官方安装指导：https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md

1）创建独立conda虚拟环境变量
conda create -n detectron python=3.6  (detectron为虚拟环境的名字,名字随便取)
2）激活环境变量
source activate detectron
3）安装相关的依赖项目：
```
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
      build-essential \
      git \
      libgoogle-glog-dev \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      libprotobuf-dev \
      openmpi-bin \
      openmpi-doc \
      protobuf-compiler \
      python-dev \
      python-pip                          
pip install --user \
      future \
      numpy \
      protobuf \
      typing \
      hypothesis
然后继续：
sudo apt-get install -y --no-install-recommends \
      libgflags-dev \
      cmake
因为caffe2已经集成到pytorch1.0中，所以直接安装pytorch:   
conda install pytorch torchvision -c pytorch
漫长等待，安装。
```
![图片标题](https://github.com/liuliu408/image/blob/master/notebook/zt2_image1.png)

注：若此处安装的时候出现solving enviroment failure,可以试试添加一些镜像源到anaconda里面去，
添加清华镜像源命令：
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
conda info
```

## 重要提示：请在anaconda下建立pytorch1.0环境
###注意pytorch1.0集成了caffe2！我们建议在anaconda下安装了pytorch1.0软件包。
注意：detectron本身是不使用pytorch1.0，而是用集成在里面的caffe2！！由于我已在anaconda3下安装了pytorch1.0环境，因而相关依赖我需要通过激活环境（source activate pytorch10）来安装，完后你就可以退出环境 （source deactivate）！直接就可以使用caffe2！因为我们是通在**.bashrc**环境变量文件中设置了路径，从而可以找到路径使用相关功能模块！ 

```
sudo chmod 777 -R /home/qiang/anaconda3   #sudo 命令对于conda是没有用的，要对conda安装目录添加sudo权限 
source activate pytorch10
conda install \
      future \
      numpy \
      protobuf
source deactivate      
等等，需要安装相关依赖就激活环境安装！！
```
https://blog.csdn.net/wishchin/article/details/79435281

>Installing Detectron===https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md
Installing caffe2     ===https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=compile

## 1. Caffe2 安装
caffe2 安装之后需要将 caffe2 添加到 PYTHONPATH 和 LD_LIBRARY_PATH 路径！
我已经通过ananconda安装了pytorch1.0包，由于官方已经在pytorch1.0中集成了Caffe2！因此就不用单独去下载、安装、编译了！
 
## （1）我的环境变量设置
### $ gedit ~/.bashrc
```
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

# added by Anaconda3 installer
export PATH="/home/qiang/anaconda3/bin:$PATH"

#echo $PYTHONPATH
export PYTHONPATH=/usr/local:$PYTHONPATH
#由于caffe2环境集成在anaconda3下的pytorch1.0中，所以一定要在环境变量中指明caffe2的路径！！！
export PYTHONPATH=/home/qiang/anaconda3/envs/pytorch10/lib/python3.6/site-packages:$PYTHONPATH
#如果是自己单独安装caffer2的话，那就需要用下面语句
#export PYTHONPATH=$PYTHONPATH:/home/qiang/caffe2/caffe2/build

#echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

export GIO_EXTRA_MODULES=/usr/lib/x86_64-linux-gnu/gio/modules/
```
### $ source  ~/.bashrc 然后使其生效

## （2）验证安装:
```
检查Caffe2是否编译安装成功
# To check if Caffe2 build was successful
$ python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

#检查Caffe2的GPU依赖是否正确，下面命令输出的GPU卡的数量必须要大于0
#否则不能使用Detectron
# To check if Caffe2 GPU build was successful
# This must print a number > 0 in order to use Detectron
$ python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```
### 正常结果如下：
```
cpu版本:
bobo@Xixi:~$ python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
Success
返回success，即为成功

gpu版本：
bobo@Xixi:~$ python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
4
返回具体数字(>0)，即为成功

qiang@qiang:~/detectron$ python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
Success

qiang@qiang:~/detectron$ python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
1

```
注意：如果此处在build目录里验证安装成功，而在根目录下验证失败，说明只是环境变量问题，添加环境变量路径


---
##2. Detectron 安装
https://github.com/facebookresearch/Detectron
###（1）下载Detectron
```
git clone https://github.com/facebookresearch/detectron $DETECTRON
若子模块未完全下载，输入以下命令直到全部下载成功：$ git submodule update --init --recursive
```
###（2）编译detectron
```
qiang@qiang:~/detectron$ make 
python setup.py develop --user
running develop
running egg_info
writing Detectron.egg-info/PKG-INFO
writing dependency_links to Detectron.egg-info/dependency_links.txt
writing top-level names to Detectron.egg-info/top_level.txt
reading manifest file 'Detectron.egg-info/SOURCES.txt'
writing manifest file 'Detectron.egg-info/SOURCES.txt'
running build_ext
building 'detectron.utils.cython_bbox' extension
creating build
creating build/temp.linux-x86_64-3.6
creating build/temp.linux-x86_64-3.6/detectron
creating build/temp.linux-x86_64-3.6/detectron/utils
gcc -pthread -B /home/qiang/anaconda3/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/qiang/anaconda3/envs/pytorch10/lib/python3.6/site-packages/numpy/core/include -I/home/qiang/anaconda3/include/python3.6m -c detectron/utils/cython_bbox.c -o build/temp.linux-x86_64-3.6/detectron/utils/cython_bbox.o -Wno-cpp
creating build/lib.linux-x86_64-3.6
creating build/lib.linux-x86_64-3.6/detectron
creating build/lib.linux-x86_64-3.6/detectron/utils
gcc -pthread -shared -B /home/qiang/anaconda3/compiler_compat -L/home/qiang/anaconda3/lib -Wl,-rpath=/home/qiang/anaconda3/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/detectron/utils/cython_bbox.o -o build/lib.linux-x86_64-3.6/detectron/utils/cython_bbox.cpython-36m-x86_64-linux-gnu.so
building 'detectron.utils.cython_nms' extension
gcc -pthread -B /home/qiang/anaconda3/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/qiang/anaconda3/envs/pytorch10/lib/python3.6/site-packages/numpy/core/include -I/home/qiang/anaconda3/include/python3.6m -c detectron/utils/cython_nms.c -o build/temp.linux-x86_64-3.6/detectron/utils/cython_nms.o -Wno-cpp
detectron/utils/cython_nms.c: In function ‘__pyx_pf_9detectron_5utils_10cython_nms_2soft_nms’:
detectron/utils/cython_nms.c:3373:34: warning: comparison between signed and unsigned integer expressions [-Wsign-compare]
       __pyx_t_11 = ((__pyx_v_pos < __pyx_v_N) != 0);
                                  ^
detectron/utils/cython_nms.c:3884:34: warning: comparison between signed and unsigned integer expressions [-Wsign-compare]
       __pyx_t_11 = ((__pyx_v_pos < __pyx_v_N) != 0);
                                  ^
gcc -pthread -shared -B /home/qiang/anaconda3/compiler_compat -L/home/qiang/anaconda3/lib -Wl,-rpath=/home/qiang/anaconda3/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/detectron/utils/cython_nms.o -o build/lib.linux-x86_64-3.6/detectron/utils/cython_nms.cpython-36m-x86_64-linux-gnu.so
copying build/lib.linux-x86_64-3.6/detectron/utils/cython_bbox.cpython-36m-x86_64-linux-gnu.so -> detectron/utils
copying build/lib.linux-x86_64-3.6/detectron/utils/cython_nms.cpython-36m-x86_64-linux-gnu.so -> detectron/utils
Creating /home/qiang/.local/lib/python3.6/site-packages/Detectron.egg-link (link to .)
Adding Detectron 0.0.0 to easy-install.pth file

Installed /home/qiang/detectron
Processing dependencies for Detectron==0.0.0
Finished processing dependencies for Detectron==0.0.0
```
如果您再次输入命令make，那就返回下面内容！
```
qiang@qiang:~/detectron$ make 
python setup.py develop --user
running develop
running egg_info
writing Detectron.egg-info/PKG-INFO
writing dependency_links to Detectron.egg-info/dependency_links.txt
writing top-level names to Detectron.egg-info/top_level.txt
reading manifest file 'Detectron.egg-info/SOURCES.txt'
writing manifest file 'Detectron.egg-info/SOURCES.txt'
running build_ext
copying build/lib.linux-x86_64-3.6/detectron/utils/cython_bbox.cpython-36m-x86_64-linux-gnu.so -> detectron/utils
copying build/lib.linux-x86_64-3.6/detectron/utils/cython_nms.cpython-36m-x86_64-linux-gnu.so -> detectron/utils
Creating /home/qiang/.local/lib/python3.6/site-packages/Detectron.egg-link (link to .)
Detectron 0.0.0 is already the active version in easy-install.pth

Installed /home/qiang/detectron
Processing dependencies for Detectron==0.0.0
Finished processing dependencies for Detectron==0.0.0
```
一共编译出下面这4个库：
```
qiang@qiang:~/detectron$ find . -name "*.so"
./build/lib.linux-x86_64-3.6/detectron/utils/cython_nms.cpython-36m-x86_64-linux-gnu.so
./build/lib.linux-x86_64-3.6/detectron/utils/cython_bbox.cpython-36m-x86_64-linux-gnu.so
./detectron/utils/cython_nms.cpython-36m-x86_64-linux-gnu.so
./detectron/utils/cython_bbox.cpython-36m-x86_64-linux-gnu.so
```
###（3）检查Detectron测试通过（比如SpatialNarrowAsOp测试）
$$ cd /home/qiang/detectron  
$ python ./detectron/tests/test_spatial_narrow_as_op.py

```
qiang@qiang:~/detectron$ python ./detectron/tests/test_spatial_narrow_as_op.py
[E init_intrinsics_check.cc:43] CPU feature avx is present on your machine, but the Caffe2 binary is not compiled with it. It means you may not get the full speed of your CPU.
[E init_intrinsics_check.cc:43] CPU feature avx2 is present on your machine, but the Caffe2 binary is not compiled with it. It means you may not get the full speed of your CPU.
[E init_intrinsics_check.cc:43] CPU feature fma is present on your machine, but the Caffe2 binary is not compiled with it. It means you may not get the full speed of your CPU.
Found Detectron ops lib: /home/qiang/anaconda3/envs/pytorch10/lib/python3.6/site-packages/torch/lib/libcaffe2_detectron_ops_gpu.so
...
----------------------------------------------------------------------
Ran 3 tests in 5.684s

OK
``` 
## 3. 编译安装COCO API
```
gemfield@caffe2:~# git clone https://github.com/cocodataset/cocoapi.git 
gemfield@caffe2:~# cd cocoapi/PythonAPI/
gemfield@caffe2:~/cocoapi/PythonAPI# make install
```

##4. 用mask-rcnn跑两个例程（使用detectron进行推断测试）
 注：根据--wts参数指定的 URL 自动下载模型，此处可以先将model_final.pkl下载下来，然后将其改为存放路径即可；根据--output-dir参数指定的路径，输出检测的可视化结果，PDF格式
        
https://github.com/facebookresearch/Detectron/blob/master/GETTING_STARTED.md
```
python tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir /tmp/detectron-visualizations \
    --image-ext jpg \
    --wts https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    demo

python tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml \
    --output-dir /tmp/detectron-visualizations \
    --image-ext jpg \
    --wts https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    demo
```
建议自己下载了model_final.pkl，放在了自定义的路径models下：~/detectron/models
模型库与基线结果：[https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md) 

```
python tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir ~/detectron/detectron-visualizations \
    --image-ext jpg \
    --wts ~/detectron/models/model_final.pkl \
    demo

python tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml \
    --output-dir ~/detectron/detectron-visualizations \
    --image-ext jpg \
    --wts ~/detectron/models/model_final.pkl \
    demo
```
命令解释：

    在上面的使用中，infer_simple.py一共使用了5个参数：
    1) --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml
    使用--cfg来指定模型的配置文件，该文件等同于solver.prototxt加上py-mask_rcnn中的配置文件；
    2) --output-dir /tmp/detectron-visualizations
    把检测结果可视化，并以pdf的格式生成在/tmp/detectron-visualizations目录中；
    3) --image-ext jpg
    寻找jpg后缀的文件；
    4) --wts https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl
    模型文件，支持http协议，这种情况下，将会下载此模型文件到本地的/tmp目录下：
    gemfield@caffe2:~/detectron/tools# ls -l 
    注意：在本命令中也会下载cfg文件中配置的预训练模型（如果使用的是http地址的话）。
    5)demo
    检测当前demo目录里jpg后缀的图片；结果最终，检测结果就以pdf的格式输出到了/tmp/detectron-visualizations目录下：

##**部分依赖安装**
```
(pytorch10) qiang@qiang:~/detectron/detectron$ python ./detectron/tests/test_spatial_narrow_as_op.py
Traceback (most recent call last):
  File "./detectron/tests/test_spatial_narrow_as_op.py", line 24, in <module>
    from caffe2.proto import caffe2_pb2
  File "/home/qiang/anaconda3/envs/pytorch10/lib/python3.6/site-packages/caffe2/proto/caffe2_pb2.py", line 6, in <module>
    from google.protobuf.internal import enum_type_wrapper
ModuleNotFoundError: No module named 'google'
(pytorch10) qiang@qiang:~/detectron/detectron$ sudo pip install protobuf
Collecting protobuf
  Using cached https://files.pythonhosted.org/packages/c2/f9/28787754923612ca9bfdffc588daa05580ed70698add063a5629d1a4209d/protobuf-3.6.1-cp36-cp36m-manylinux1_x86_64.whl
Requirement already satisfied: setuptools in /home/qiang/anaconda3/lib/python3.6/site-packages (from protobuf) (39.1.0)
Requirement already satisfied: six>=1.9 in /home/qiang/anaconda3/lib/python3.6/site-packages (from protobuf) (1.11.0)
Installing collected packages: protobuf
Successfully installed protobuf-3.6.1


(pytorch10) qiang@qiang:~/detectron/detectron$ python ./detectron/tests/test_spatial_narrow_as_op.py
Traceback (most recent call last):
  File "./detectron/tests/test_spatial_narrow_as_op.py", line 25, in <module>
    from caffe2.python import core
  File "/home/qiang/anaconda3/envs/pytorch10/lib/python3.6/site-packages/caffe2/python/core.py", line 9, in <module>
    from past.builtins import basestring
ModuleNotFoundError: No module named 'past'
(pytorch10) qiang@qiang:~/detectron/detectron$ pip install future
Collecting future
  Downloading https://files.pythonhosted.org/packages/90/52/e20466b85000a181e1e144fd8305caf2cf475e2f9674e797b222f8105f5f/future-0.17.1.tar.gz (829kB)
    100% |████████████████████████████████| 829kB 59kB/s 
Building wheels for collected packages: future
  Running setup.py bdist_wheel for future ... done
  Stored in directory: /home/qiang/.cache/pip/wheels/0c/61/d2/d6b7317325828fbb39ee6ad559dbe4664d0896da4721bf379e
Successfully built future
Installing collected packages: future
Successfully installed future-0.17.1


运行测试：python -m caffe2.python.operator_test.relu_op_test

显示：
WARNING:root:This caffe2 python run does not have GPU support. Will run in CPU only mode.
WARNING:root:Debug message: No module named caffe2_pybind11_state_gpu
Traceback (most recent call last):
  File "/home/wishchin/anaconda2/lib/python2.7/runpy.py", line 174, in _run_module_as_main
    "__main__", fname, loader, pkg_name)
  File "/home/wishchin/anaconda2/lib/python2.7/runpy.py", line 72, in _run_code
    exec code in run_globals
  File "/home/wishchin/CNN/caffe2/build/caffe2/python/operator_test/relu_op_test.py", line 22, in <module>
    from hypothesis import given
ImportError: No module named hypothesis

解决：
conda install \
      flask \
      graphviz \
      hypothesis \
      jupyter \
      matplotlib \
      pydot python-nvd3 \
      pyyaml \
      requests \
      scikit-image \
      scipy \
      setuptools \
      tornado

### which python
/home/qiang/anaconda3/envs/pytorch10/bin/python
### echo $PYTHONPATH:
:/home/qiang/anaconda3/lib/python3.6/site-packages
### echo $PATH
/home/qiang/anaconda3/envs/pytorch10/bin:
/home/qiang/anaconda3/condabin:
/home/qiang/anaconda3/bin:/usr/local/cuda-.0/bin:/home/qiang/bin:
/home/qiang/.local/bin:
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

```

查看环境路径
```
(detectron) bobo@Xixi:~/liuq/detectron$ python
Python 3.7.2 (default, Dec 29 2018, 06:19:36) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys
>>> sys.path 
>>> print("\n".join(sys.path))
/home/bobo/anaconda3/envs/detectron/lib/python37.zip
/home/bobo/anaconda3/envs/detectron/lib/python3.7
/home/bobo/anaconda3/envs/detectron/lib/python3.7/lib-dynload
/home/bobo/.local/lib/python3.7/site-packages
/home/bobo/.local/lib/python3.7/site-packages/pycocotools-2.0-py3.7-linux-x86_64.egg
/home/bobo/Hdisk/ljc/detectron
/home/bobo/anaconda3/envs/detectron/lib/python3.7/site-packages/matplotlib-3.0.2-py3.7-linux-x86_64.egg
/home/bobo/anaconda3/envs/detectron/lib/python3.7/site-packages/Cython-0.29.4-py3.7-linux-x86_64.egg
/home/bobo/anaconda3/envs/detectron/lib/python3.7/site-packages/python_dateutil-2.7.5-py3.7.egg
/home/bobo/anaconda3/envs/detectron/lib/python3.7/site-packages/pyparsing-2.3.1-py3.7.egg
/home/bobo/anaconda3/envs/detectron/lib/python3.7/site-packages/kiwisolver-1.0.1-py3.7-linux-x86_64.egg
/home/bobo/anaconda3/envs/detectron/lib/python3.7/site-packages/cycler-0.10.0-py3.7.egg
/home/bobo/anaconda3/envs/detectron/lib/python3.7/site-packages
/home/bobo/anaconda3/envs/detectron/lib/python3.7/site-packages/pycocotools-2.0-py3.7-linux-x86_64.egg

P.S.其他系统命令：
sudo uname --s 显示内核名字s
sudo uname --r 显示内核版本
sudo uname --n 显示网络主机名
sudo uname --p 显示cpu
lspci |grep -i vga（查看显卡型号）

(detectron) bobo@Xixi:~$ sudo uname --n 
Xixi
(detectron) bobo@Xixi:~$ sudo uname --r
4.15.0-43-generic
(detectron) bobo@Xixi:~$ sudo uname --s
Linux
(detectron) bobo@Xixi:~$ sudo uname --p 
x86_64
(detectron) bobo@Xixi:~$ lspci |grep -i vga
02:00.0 VGA compatible controller: NVIDIA Corporation Device 1b06 (rev a1)
03:00.0 VGA compatible controller: NVIDIA Corporation Device 1b06 (rev a1)
06:00.0 VGA compatible controller: ASPEED Technology, Inc. ASPEED Graphics Family (rev 30)
82:00.0 VGA compatible controller: NVIDIA Corporation Device 1b06 (rev a1)
83:00.0 VGA compatible controller: NVIDIA Corporation Device 1b06 (rev a1)
```

1. config.py 给出了 Detectron 的默认参数，其位于 detectron/core/config.py
https://blog.csdn.net/zziahgf/article/details/79652946

