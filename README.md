![image](./assets/sophgo_chip.png)

# MiniCPM

本项目实现BM1684X部署语言大模型[MiniCPM-2B](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16-llama-format)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

下文中默认是PCIE环境；如果是SoC环境，按提示操作即可。

# 目录说明
```
.
├── assets
│   └── sophgo_chip.png
├── compile
│   ├── compile.sh                       #用来编译TPU模型的脚本
│   ├── export_onnx.py                   #用来导出onnx的脚本
│   └── files                            #用于替换原模型的文件
├── models
│   └── MiniCPM-2B_int8_1dev_512.bmodel
├── python_demo
│   ├── chat.cpp                         #MiniCPM c++代码文件
│   ├── CMakeLists.txt
│   ├── pipeline.py                      #pybind 后的python主程序
│   └── README.md
├── README.md                            #使用说明
├── requirements.txt                     #需要使用的python wheel包
└── support
    └── token_config                     #编译相关模型文件

```
----------------------------

# 【阶段一】模型编译

## 注意点
* 模型编译必须要在docker内完成，无法在docker外操作

### 步骤一：模型下载
虽然MiniCPM模型允许商业开源，但是模型下载需要想Meta提交使用申请，因此测试模型时可以使用我们已经下载好的模型
```bash
pip3 install dfss
# minicpm-2B
python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/minicpm-2b-torch.zip
unzip minicpm-2b-torch.zip
```

### 步骤二：下载docker

下载docker，启动容器，如下：

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

### 步骤三：下载TPU-MLIR代码并编译

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```
* PS：重新进入docker环境并且需要编译模型时，必须在此路径下执行上述`source ./envsetup.sh` 和 `./build.sh`才能完成后续模型编译。

### 步骤四：对齐模型环境

``` shell
pip install -r requirements.txt
cp ./compile/files/minicpm-2b-chat-llama/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
```

* PS：不一定是/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py这个路径，建议替换前先pip show transformers查看一下

### 步骤五：生成onnx文件

``` shell
cd compile
python export_onnx.py --model_path your_model_path --seq_length 512
```

* PS1：your_model_path 指的是原模型下载后的地址, 如:"../../torch2onnx/minicpm-2b-chat-llama", 可以根据需要选择使用7b模型还是13b模型。
* PS2：如果你想要debug，而不是一下子生成完成全部的onnx模型，可以将240行的num_layers改成1, 结合233行的函数对比单个block情况下是否可以和
* PS3：默认导出sequence length为512的模型

### 步骤六：生成bmodel文件

生成单芯模型

``` shell
./compile.sh --mode int8 --name MiniCPM-2B # same as int4
```

* PS1：编译完成后最终会在compile路径下生成名为MiniCPM-{X}B_{Y}_{Z}dev.bmodel,其中X为2，Y为`compile.sh`时选择的`mode`的数据类型,Z为推理的芯片数量(如果不指定num_device, 会省略{Z}dev的部分)
* PS2：生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left
* PS3：如果想要编译minicpm-2b，则--name必须为minicpm-2b
* PS4：目前给定的lib_pcie和lib_soc部分仅包含单芯的动态库，多芯部分会在后续更新

----------------------------

# 阶段二：python调用pipeline程序执行推理并进行输出


## 编译库文件
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

## python demo
```
python3 pipeline.py --model_path MiniCPM-2B_int8_1dev_512.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```