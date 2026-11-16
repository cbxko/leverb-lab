# LeVERB Lab

LeVERB Lab 是一个面向人形机器人的视觉-语言-动作（VLA）项目，独立实现了 [LeVERB 论文](https://arxiv.org/abs/2506.13751)中的上层模型 LeVERB-VL。

模型根据第一人称画面、第三人称画面和语言指令生成一个动作向量。解码器结合这个向量与机器人当前的姿态，预测未来约一秒的身体运动。例如，给出「走向沙发」的指令，模型要预测机器人接下来会朝哪个方向移动、身体姿态会怎样变化。

本项目从公开的 [LeVERB-Bench 数据集](https://huggingface.co/datasets/ember-lab-berkeley/LeVERB-Bench-Dataset)中选取了 570 条示范，用于训练和离线验证。验证集上的最低轨迹预测 MSE 为 **0.01183**，比「保持当前姿态不动」的基线低 **16.8%**。

当前实现限于上层模型与离线轨迹预测，暂未实现低层控制器 LeVERB-A 或真机部署。
