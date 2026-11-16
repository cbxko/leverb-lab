# LeVERB Lab

LeVERB Lab 是一个面向人形机器人的视觉-语言-动作（VLA）项目，独立实现了 [LeVERB 论文](https://arxiv.org/abs/2506.13751)中的上层模型 LeVERB-VL。

模型根据第一人称画面、第三人称画面和语言指令生成一个动作向量。解码器结合这个向量与机器人当前的姿态，预测未来约一秒的身体运动。例如，给出「走向沙发」的指令，模型要预测机器人接下来会朝哪个方向移动、身体姿态会怎样变化。

本项目从公开的 [LeVERB-Bench 数据集](https://huggingface.co/datasets/ember-lab-berkeley/LeVERB-Bench-Dataset)中选取了 570 条示范，用于训练和离线验证。验证集上的最低轨迹预测 MSE 为 **0.01183**，比「保持当前姿态不动」的基线低 **16.8%**。

当前实现限于上层模型与离线轨迹预测，暂未实现低层控制器 LeVERB-A 或真机部署。

## 项目结构

```text
leverb-lab/
├── train.py                              # 训练入口：前向检查、短训练和按轮训练
├── requirements.txt                      # Python 依赖
└── src/
    ├── data_processing/
    │   ├── prepare_vl_sample.py           # 生成试样、划分数据集、批量切分轨迹
    │   └── verify_vl_sample.py            # 检查运动目标能否还原原始轨迹
    ├── data_loading/
    │   └── vl_dataset.py                 # 按索引读取双视角图像、指令和运动数据，并组批
    ├── models/
    │   ├── siglip_encoder.py             # 提取双视角图像和文本特征，权重冻结
    │   ├── vl_transformer.py             # 融合图文特征，输出动作向量的先验分布
    │   ├── kinematics_encoder.py         # 根据真实未来轨迹提供后验参数
    │   ├── residual_latent.py            # 构造后验，从先验或后验生成动作向量
    │   ├── kinematics_decoder.py         # 根据当前姿态和动作向量预测未来运动
    │   └── leverb_vl.py                  # 组装模型，区分训练和验证的前向路径
    └── training/
        ├── losses.py                    # 轨迹重建损失和先验、后验之间的 KL 损失
        └── trainer.py                   # 参数更新、全量验证、早停和权重保存
```

数据先按任务名划分，再把每条示范切成训练窗口。数据加载器按帧索引取出双视角图像，配上语言指令、当前姿态和未来轨迹。

代码仓库不包含视频、处理结果和模型权重。训练时，原始示范从 `data/raw/LeVERB-Bench-Dataset/` 读取，处理结果从 `data/processed/vl_batch/` 读取，训练记录和权重默认写入 `runs/`。

## LeVERB-VL 模型模块

本仓库将 LeVERB-VL 拆成 **5 个功能模块**，另有一个总模型负责组装。

### 1. SigLIP 图文编码器

代码：[`siglip_encoder.py`](src/models/siglip_encoder.py) · `SigLIPEncoder`

将当前的第一人称图像、第三人称图像和文本指令分别编码为一个 768 维特征。两路图像复用同一个视觉编码器，图像和文本编码器的权重在训练中保持冻结。

### 2. 视觉语言先验

代码：[`vl_transformer.py`](src/models/vl_transformer.py) · `VLTransformerPrior`

将三个图文特征作为三个 token 送入 Transformer，再经 MLP 输出 256 维高斯先验的均值和对数方差。这个分布只依赖当前画面和指令。

### 3. 运动学编码器

代码：[`kinematics_encoder.py`](src/models/kinematics_encoder.py) · `KinematicsEncoder`

训练时读取示范中真实的未来 10 步轨迹，展平后交给 MLP，输出均值修正量和后验对数方差。均值修正量用于补充图文先验难以确定的运动细节。

### 4. 残差潜变量

代码：[`residual_latent.py`](src/models/residual_latent.py) · `ResidualLatent`

将先验均值与修正量相加得到后验均值，后验对数方差直接使用运动学编码器的输出。训练时从后验重参数化采样，得到 256 维动作向量 $z_t$。离线验证时，当前实现默认取先验均值。

### 5. 运动学解码器

代码：[`kinematics_decoder.py`](src/models/kinematics_decoder.py) · `KinematicsDecoder`

将动作向量 $z_t$ 与 38 维当前身体状态拼接，经 MLP 输出未来 10 步、每步 45 维的运动变化，包括骨盆平移、相对旋转和身体部位的位置变化。

### 总模型

代码：[`leverb_vl.py`](src/models/leverb_vl.py) · `LeVERBVL`

组装以上模块，管理训练的后验分支和离线验证的先验分支。训练所用的轨迹重建 MSE 和先验、后验之间的 KL 损失定义在 [`losses.py`](src/training/losses.py)。

论文还使用判别器来对齐有图像与无图像的示范。本仓库只使用带图像的示范，未实现该模块。
