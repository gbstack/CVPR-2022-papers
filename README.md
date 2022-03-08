
# CVPR2022 Papers (Papers/Codes/Demos)



## 分类目录：

### [1. 检测](#detection)


### [2. 分割(Segmentation)](#Segmentation)

### [3. 图像处理(Image Processing)](#ImageProcessing)

### [4. 估计(Estimation)](#Estimation)

### [5. 图像&视频检索/视频理解(Image&Video Retrieval/Video Understanding)](#)

### [6. 人脸(Face)](#Face)


### [7. 三维视觉(3D Vision)](#3DVision)


### [8. 目标跟踪(Object Tracking)](#ObjectTracking)

### [9. 医学影像(Medical Imaging)](#MedicalImaging)

### [10. 文本检测/识别(Text Detection/Recognition)](#TDR)

### [11. 遥感图像(Remote Sensing Image)](#RSI)

### [12. GAN/生成式/对抗式(GAN/Generative/Adversarial)](#GAN)

### [13. 图像生成/合成(Image Generation/Image Synthesis)](#IGIS)


### [14. 场景图(Scene Graph](#SG)


### [15. 视觉定位(Visual Localization)](#VisualLocalization)

### [16. 视觉推理/视觉问答(Visual Reasoning/VQA)](#VisualReasoning)

### [17. 图像分类(Image Classification)](#ImageClassification)

### [18. 神经网络结构设计(Neural Network Structure Design)](#NNS)


### [19. 模型压缩(Model Compression)](#ModelCompression)


### [20. 模型训练/泛化(Model Training/Generalization)](#ModelTraining)


### [21. 模型评估(Model Evaluation)](#ModelEvaluation)

### [22. 数据处理(Data Processing)](#DataProcessing)


### [23. 主动学习(Active Learning)](#ActiveLearning)

### [24. 小样本学习/零样本学习(Few-shot/Zero-shot Learning)](#Few-shotLearning)

### [25. 持续学习(Continual Learning/Life-long Learning)](#ContinualLearning)

### [26. 迁移学习/domain/自适应(Transfer Learning/Domain Adaptation)](#domain)

### [27. 度量学习(Metric Learning)](#MetricLearning)

### [28. 对比学习(Contrastive Learning)](#ContrastiveLearning)

### [29. 增量学习(Incremental Learning)](#IncrementalLearning)

### [30. 强化学习(Reinforcement Learning)](#RL)

### [31. 元学习(Meta Learning)](#MetaLearning)

### [32. 多模态学习(Multi-Modal Learning)](#MMLearning)


### [33. 视觉预测(Vision-based Prediction)](#Vision-basedPrediction)

### [34. 数据集(Dataset)](#Dataset)

### [35. 机器人(Robotic)](#Robotic)

### [36. 自监督学习/半监督学习](#self-supervisedlearning)

<br><br>


    
<a name="detection"></a>
## 检测
<br>
        
### 图像目标检测(2D Object Detection)

**Localization Distillation for Dense Object Detection(密集对象检测的定位蒸馏)**<br>
*keywords: Bounding Box Regression, Localization Quality Estimation, Knowledge Distillation*<br>
[paper](https://arxiv.org/abs/2102.12252) | [code](https://github.com/HikariTJU/LD)<br>
<br>

### 视频目标检测(Video Object Detection)

**Unsupervised Activity Segmentation by Joint Representation Learning and Online Clustering(通过联合表示学习和在线聚类进行无监督活动分割)**<br>
[paper](https://arxiv.org/abs/2105.13353)<br>
<br>

### 3D目标检测(3D object detection)

**A Versatile Multi-View Framework for LiDAR-based 3D Object Detection with Guidance from Panoptic Segmentation(在全景分割的指导下，用于基于 LiDAR 的 3D 对象检测的多功能多视图框架)**<br>
*keywords: 3D Object Detection with Point-based Methods, 3D Object Detection with Grid-based Methods, Cluster-free 3D Panoptic Segmentation, CenterPoint 3D Object Detection*<br>
[paper](https://arxiv.org/abs/2203.02133)<br>
<br>
**Pseudo-Stereo for Monocular 3D Object Detection in Autonomous Driving(自动驾驶中用于单目 3D 目标检测的伪立体)**<br>
*keywords: Autonomous Driving, Monocular 3D Object Detection*<br>
[paper](https://arxiv.org/abs/2203.02112) | [code](https://github.com/revisitq/Pseudo-Stereo-3D)<br>
<br>

<a name="Segmentation"></a>
## 分割(Segmentation)
<br>
        
### 车道线检测(Lane Detection)

**Rethinking Efficient Lane Detection via Curve Modeling(通过曲线建模重新思考高效车道检测)**<br>
*keywords: Segmentation-based Lane Detection, Point Detection-based Lane Detection, Curve-based Lane Detection, autonomous driving*<br>
[paper](https://arxiv.org/abs/2203.02431) | [code](https://github.com/voldemortX/pytorch-auto-drive)<br>
<br>

### 全景分割(Panoptic Segmentation)

**Bending Reality: Distortion-aware Transformers for Adapting to Panoramic Semantic Segmentation(弯曲现实：适应全景语义分割的失真感知Transformer)**<br>
*keywords: Semanticand panoramic segmentation, Unsupervised domain adaptation, Transformer*<br>
[paper](https://arxiv.org/abs/2203.01452) | [code](https://github.com/jamycheung/Trans4PASS)<br>
<br>

### 语义分割(Semantic Segmentation)

**ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation(让自我训练更好地用于半监督语义分割)**<br>
*keywords: Semi-supervised learning, Semantic segmentation, Uncertainty estimation*<br>
[paper](https://arxiv.org/abs/2106.05095) | [code](https://github.com/LiheYoung/ST-PlusPlus)<br>
<br>
**Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation(弱监督语义分割的类重新激活图)**<br>
[paper](https://arxiv.org/pdf/2203.00962.pdf) | [code](https://github.com/zhaozhengChen/ReCAM)<br>
<br>

### 实例分割(Instance Segmentation)

**Efficient Video Instance Segmentation via Tracklet Query and Proposal(通过 Tracklet Query 和 Proposal 进行高效的视频实例分割)**<br>
[paper](https://arxiv.org/abs/2203.01853)<br>
<br>
**SoftGroup for 3D Instance Segmentation on Point Clouds(用于点云上的 3D 实例分割)**<br>
*keywords: 3D Vision, Point Clouds, Instance Segmentation*<br>
[paper](https://arxiv.org/abs/2203.01509) | [code](https://github.com/thangvubk/SoftGroup.git)<br>
<br>

<a name="Estimation"></a>
## 估计(Estimation)
<br>
        
### 姿态估计(Human Pose Estimation)

**Learning Local-Global Contextual Adaptation for Multi-Person Pose Estimation(学习用于多人姿势估计的局部-全局上下文适应)**<br>
*keywords: Top-Down Pose Estimation(从上至下姿态估计), Limb-based Grouping, Direct Regression*<br>
[paper](https://arxiv.org/pdf/2109.03622.pdf)<br>
<br>
**MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video(用于视频中 3D 人体姿势估计的 Seq2seq 混合时空编码器)**<br>
[paper](https://arxiv.org/pdf/2203.00859.pdf)<br>
<br>

### 深度估计(Depth Estimation)

**ITSA: An Information-Theoretic Approach to Automatic Shortcut Avoidance and Domain Generalization in Stereo Matching Networks(立体匹配网络中自动避免捷径和域泛化的信息论方法)**<br>
*keywords: Learning-based Stereo Matching Networks, Single Domain Generalization, Shortcut Learning*<br>
[paper](https://arxiv.org/pdf/2201.02263.pdf)<br>
<br>
**Attention Concatenation Volume for Accurate and Efficient Stereo Matching(用于精确和高效立体匹配的注意力连接体积)**<br>
*keywords: Stereo Matching, cost volume construction, cost aggregation*<br>
[paper](https://arxiv.org/pdf/2203.02146.pdf) | [code](https://github.com/gangweiX/ACVNet)<br>
<br>
**Occlusion-Aware Cost Constructor for Light Field Depth Estimation(光场深度估计的遮挡感知成本构造函数)**<br>
[paper](https://arxiv.org/pdf/2203.01576.pdf) | [code](https://github.com/YingqianWang/OACC- Net)<br>
<br>
**NeW CRFs: Neural Window Fully-connected CRFs for Monocular Depth Estimation(用于单目深度估计的神经窗口全连接 CRF)**<br>
*keywords: Neural CRFs for Monocular Depth*<br>
[paper](https://arxiv.org/pdf/2203.01502.pdf)<br>
<br>

<a name="ImageProcessing"></a>
## 图像处理(Image Processing)
<br>
        
### 深度估计(Depth Estimation)

**OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion(通过几何感知融合进行 360 度单目深度估计)**<br>
*keywords: monocular depth estimation(单目深度估计),transformer*<br>
[paper](https://arxiv.org/abs/2203.00838)<br>
<br>

### 超分辨率(Super Resolution)

**HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging(光谱压缩成像的高分辨率双域学习)**<br>
*keywords: HSI Reconstruction, Self-Attention Mechanism,  Image Frequency Spectrum Analysis*<br>
[paper](https://arxiv.org/pdf/2203.02149.pdf)<br>
<br>

### 图像复原/图像增强/图像重建(Image Restoration/Image Reconstruction)

**Event-based Video Reconstruction via Potential-assisted Spiking Neural Network(通过电位辅助尖峰神经网络进行基于事件的视频重建)**<br>
[paper](https://arxiv.org/pdf/2201.10943.pdf)<br>
<br>

### 图像去噪/去模糊/去雨去雾(Image Denoising)

**E-CIR: Event-Enhanced Continuous Intensity Recovery(事件增强的连续强度恢复)**<br>
*keywords: Event-Enhanced Deblurring, Video Representation*<br>
[paper](https://arxiv.org/abs/2203.01935) | [code](https://github.com/chensong1995/E-CIR)<br>
<br>

### 图像编辑/图像修复(Image Edit/Inpainting)

**HairCLIP: Design Your Hair by Text and Reference Image(通过文本和参考图像设计你的头发)**<br>
*keywords: Language-Image Pre-Training (CLIP), Generative Adversarial Networks*<br>
[paper](https://arxiv.org/abs/2112.05142)<br>
<br>
**Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding(增量transformer结构增强图像修复与掩蔽位置编码)**<br>
*keywords: Image Inpainting, Transformer, Image Generation*<br>
[paper](https://arxiv.org/abs/2203.00867) | [code](https://github.com/DQiaole/ZITS_inpainting)<br>
<br>

### 图像翻译(Image Translation)

**Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks(探索图像到图像翻译任务中对比学习的补丁语义关系)**<br>
*keywords: image translation, knowledge transfer,Contrastive learning*<br>
[paper](https://arxiv.org/pdf/2203.01532.pdf)<br>
<br>

<a name="Face"></a>
## 人脸(Face)
<br>
        
### 风格迁移(Style Transfer)

**CLIPstyler: Image Style Transfer with a Single Text Condition(具有单一文本条件的图像风格转移)**<br>
*keywords: Style Transfer, Text-guided synthesis, Language-Image Pre-Training (CLIP)*<br>
[paper](https://arxiv.org/abs/2112.00374)<br>
<br>

### 人脸识别/检测(Facial Recognition/Detection)

**An Efficient Training Approach for Very Large Scale Face Recognition(一种有效的超大规模人脸识别训练方法)**<br>
[paper](https://arxiv.org/pdf/2105.10375.pdf) | [code](https://github.com/tiandunx/FFC)<br>
<br>

### 人脸生成/合成/重建/编辑(Face Generation/Face Synthesis/Face Reconstruction/Face Editing)

**Sparse to Dense Dynamic 3D Facial Expression Generation(稀疏到密集的动态 3D 面部表情生成)**<br>
*keywords: Facial expression generation, 4D face generation, 3D face modeling*<br>
[paper](https://arxiv.org/pdf/2105.07463.pdf)<br>
<br>

### 人脸伪造/反欺骗(Face Forgery/Face Anti-Spoofing)

**Voice-Face Homogeneity Tells Deepfake**<br>
[paper](https://arxiv.org/abs/2203.02195) | [code](https://github.com/xaCheng1996/VFD)<br>
<br>

<a name="ObjectTracking"></a>
## 目标跟踪(Object Tracking)
<br>
        
### 人脸伪造/反欺骗(Face Forgery/Face Anti-Spoofing)

**Protecting Celebrities with Identity Consistency Transformer(使用身份一致性transformer保护名人)**<br>
[paper](https://arxiv.org/abs/2203.01318)<br>
<br>
**TCTrack: Temporal Contexts for Aerial Tracking(空中跟踪的时间上下文)**<br>
[paper](https://arxiv.org/abs/2203.01885) | [code](https://github.com/vision4robotics/TCTrack)<br>
<br>
**Beyond 3D Siamese Tracking: A Motion-Centric Paradigm for 3D Single Object Tracking in Point Clouds(超越 3D 连体跟踪：点云中 3D 单对象跟踪的以运动为中心的范式)**<br>
*keywords: Single Object Tracking, 3D Multi-object Tracking / Detection, Spatial-temporal Learning on Point Clouds*<br>
[paper](https://arxiv.org/abs/2203.01730)<br>
<br>

<a name="ImageRetrieval"></a>
## 图像&视频检索/视频理解(Image&Video Retrieval/Video Understanding)
<br>
        
### 人脸伪造/反欺骗(Face Forgery/Face Anti-Spoofing)

**Correlation-Aware Deep Tracking(相关感知深度跟踪)**<br>
[paper](https://arxiv.org/abs/2203.01666)<br>
<br>
**BEVT: BERT Pretraining of Video Transformers(视频Transformer的 BERT 预训练)**<br>
*keywords: Video understanding, Vision transformers, Self-supervised representation learning, BERT pretraining*<br>
[paper](https://arxiv.org/abs/2112.01529) | [code](https://github.com/xyzforever/BEVT)<br>
<br>

### 行为识别/动作识别/检测/分割/定位(Action/Activity Recognition)

**Colar: Effective and Efficient Online Action Detection by Consulting Exemplars(通过咨询示例进行有效且高效的在线动作检测)**<br>
*keywords: Online action detection(在线动作检测)*<br>
[paper](https://arxiv.org/pdf/2203.01057.pdf)<br>
<br>

<a name="MedicalImaging"></a>
## 医学影像(Medical Imaging)
<br>
        
### 图像/视频字幕(Image/Video Caption)

**X -Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning(使用 Transformer 进行 3D 密集字幕的跨模式知识迁移)**<br>
[paper](https://arxiv.org/pdf/2203.00843.pdf)<br>
<br>

<a name="GAN"></a>
## GAN/生成式/对抗式(GAN/Generative/Adversarial)
<br>
        
### 图像/视频字幕(Image/Video Caption)

**Temporal Context Matters: Enhancing Single Image Prediction with Disease Progression Representations(时间上下文很重要：使用疾病进展表示增强单图像预测)**<br>
*keywords: Self-supervised Transformer, Temporal modeling of disease progression*<br>
[paper](https://arxiv.org/abs/2203.01933)<br>
<br>

<a name="None"></a>
## 图像生成/图像合成(Image Generation/Image Synthesis)
<br>
        
### 图像/视频字幕(Image/Video Caption)

**Label-Only Model Inversion Attacks via Boundary Repulsion(通过边界排斥的仅标签模型反转攻击)**<br>
[paper](https://arxiv.org/pdf/2203.01925.pdf)<br>
<br>
**3D Shape Variational Autoencoder Latent Disentanglement via Mini-Batch Feature Swapping for Bodies and Faces(基于小批量特征交换的三维形状变化自动编码器潜在解纠缠)**<br>
[paper](https://arxiv.org/pdf/2111.12448.pdf) | [code](https://github.com/simofoti/3DVAE-SwapDisentangled)<br>
<br>
**Interactive Image Synthesis with Panoptic Layout Generation(具有全景布局生成的交互式图像合成)**<br>
[paper](https://arxiv.org/abs/2203.02104)<br>
<br>
**Polarity Sampling: Quality and Diversity Control of Pre-Trained Generative Networks via Singular Values(极性采样：通过奇异值对预训练生成网络的质量和多样性控制)**<br>
[paper](https://arxiv.org/abs/2203.01993)<br>
<br>
**Autoregressive Image Generation using Residual Quantization(使用残差量化的自回归图像生成)**<br>
[paper](https://arxiv.org/abs/2203.01941) | [code](https://github.com/kakaobrain/rq-vae-transformer)<br>
<br>

<a name="3DVision"></a>
## 三维视觉(3D Vision)
<br>
        
### 视图合成(View Synthesis)

**X -Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning(使用 Transformer 进行 3D 密集字幕的跨模式知识迁移)**<br>
[paper](https://arxiv.org/pdf/2203.00843.pdf)<br>
<br>

### 点云(Point Cloud)

**A Unified Query-based Paradigm for Point Cloud Understanding(一种基于统一查询的点云理解范式)**<br>
[paper](https://arxiv.org/pdf/2203.01252.pdf)<br>
<br>
**CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding(用于 3D 点云理解的自监督跨模态对比学习)**<br>
*keywords: Self-Supervised Learning, Contrastive Learning, 3D Point Cloud, Representation Learning, Cross-Modal Learning*<br>
[paper](https://arxiv.org/abs/2203.00680) | [code](http://github.com/MohamedAfham/CrossPoint)<br>
<br>

### 三维重建(3D Reconstruction)

**H4D: Human 4D Modeling by Learning Neural Compositional Representation(通过学习神经组合表示进行人体 4D 建模)**<br>
*keywords: 4D Representation(4D 表征),Human Body Estimation(人体姿态估计),Fine-grained Human Reconstruction(细粒度人体重建)*<br>
[paper](https://arxiv.org/pdf/2203.01247.pdf)<br>
<br>

### 场景重建/新视角合成(Novel View Synthesis)

**CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields(文本和图像驱动的神经辐射场操作)**<br>
*keywords: NeRF,  Image Generation and Manipulation, Language-Image Pre-Training (CLIP)*<br>
[paper](https://arxiv.org/abs/2112.05139) | [code](https://cassiepython.github.io/clipnerf/)<br>
<br>

<a name="ModelCompression"></a>
## 模型压缩(Model Compression)
<br>
        
### 场景重建/新视角合成(Novel View Synthesis)

**Point-NeRF: Point-based Neural Radiance Fields(基于点的神经辐射场)**<br>
[paper](https://arxiv.org/pdf/2201.08845.pdf) | [code](https://github.com/Xharlie/pointnerf)<br>
<br>

<a name="NNS"></a>
## 神经网络结构设计(Neural Network Structure Design)
<br>
        
### 量化(Quantization)

**BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning(学习探索样本关系以进行鲁棒表征学习)**<br>
*keywords: sample relationship, data scarcity learning, Contrastive Self-Supervised Learning, long-tailed recognition, zero-shot learning, domain generalization, self-supervised learning*<br>
[paper](https://arxiv.org/abs/2203.01522) | [code](https://github.com/zhihou7/BatchFormer)<br>
<br>

### CNN

**A ConvNet for the 2020s**<br>
[paper](https://arxiv.org/abs/2201.03545) | [code](https://github.com/facebookresearch/ConvNeXt)<br>
<br>

### Transformer

**Mobile-Former: Bridging MobileNet and Transformer(连接 MobileNet 和 Transformer)**<br>
*keywords: Light-weight convolutional neural networks(轻量卷积神经网络),Combination of CNN and ViT*<br>
[paper](https://arxiv.org/abs/2108.05895)<br>
<br>

### 神经网络架构搜索(NAS)

**β-DARTS: Beta-Decay Regularization for Differentiable Architecture Search(可微架构搜索的 Beta-Decay 正则化)**<br>
[paper](https://arxiv.org/abs/2203.01665)<br>
<br>

<a name="DataProcessing"></a>
## 数据处理(Data Processing)
<br>
        
### MLP

**An Image Patch is a Wave: Quantum Inspired Vision MLP(图像补丁是波浪：量子启发的视觉 MLP)**<br>
[paper](https://arxiv.org/abs/2111.12294) | [code](https://github.com/huawei-noah/CV-Backbones/tree/master/wavemlp_pytorch)<br>
<br>

### 数据增广(Data Augmentation)

**3D Common Corruptions and Data Augmentation(3D 常见损坏和数据增强)**<br>
*keywords: Data Augmentation, Image restoration, Photorealistic image synthesis*<br>
[paper](https://arxiv.org/abs/2203.01441)<br>
<br>

<a name="ModelTraining"></a>
## 模型训练/泛化(Model Training/Generalization)
<br>
        
### 异常检测(Anomaly Detection)

**Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection(用于异常检测的自监督预测卷积注意力块)(论文暂未上传)**<br>
[paper](https://arxiv.org/abs/2111.09099) | [code](https://github.com/ristea/sspcab)<br>
<br>
**CAFE: Learning to Condense Dataset by Aligning Features(通过对齐特征学习压缩数据集)**<br>
*keywords: dataset condensation, coreset selection, generative models*<br>
[paper](https://arxiv.org/pdf/2203.01531.pdf) | [code](https://github.com/kaiwang960112/CAFE)<br>
<br>
**The Devil is in the Margin: Margin-based Label Smoothing for Network Calibration(魔鬼在边缘：用于网络校准的基于边缘的标签平滑)**<br>
[paper](https://arxiv.org/abs/2111.15430) | [code](https://github.com/by-liu/MbLS)<br>
<br>
**DN-DETR: Accelerate DETR Training by Introducing Query DeNoising(通过引入查询去噪加速 DETR 训练)**<br>
*keywords: Detection Transformer*<br>
[paper](https://arxiv.org/abs/2203.01305) | [code](https://github.com/FengLi-ust/DN-DETR)<br>
<br>

<a name="MMLearning"></a>
## 多模态学习(Multi-Modal Learning)
<br>
        
### 长尾分布(Long-Tailed Distribution)

**Targeted Supervised Contrastive Learning for Long-Tailed Recognition(用于长尾识别的有针对性的监督对比学习)**<br>
*keywords: Long-Tailed Recognition(长尾识别), Contrastive Learning(对比学习)*<br>
[paper](https://arxiv.org/pdf/2111.13998.pdf)<br>
<br>

### 视觉语言（Vision-language Representation Learning）

**HairCLIP: Design Your Hair by Text and Reference Image(通过文本和参考图像设计你的头发)**<br>
*keywords: Language-Image Pre-Training (CLIP), Generative Adversarial Networks*<br>
[paper](https://arxiv.org/abs/2112.05142)<br>
<br>
**CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields(文本和图像驱动的神经辐射场操作)**<br>
*keywords: NeRF,  Image Generation and Manipulation, Language-Image Pre-Training (CLIP)*<br>
[paper](https://arxiv.org/abs/2112.05139) | [code](https://cassiepython.github.io/clipnerf/)<br>
<br>

<a name="None"></a>
## 场景图(Scene Graph)
<br>
        
### 视觉语言（Vision-language Representation Learning）

**Vision-Language Pre-Training with Triple Contrastive Learning(三重对比学习的视觉语言预训练)**<br>
*keywords: Vision-language representation learning, Contrastive Learning*<br>
[paper](https://arxiv.org/abs/2202.10401) | [code](https://github.com/uta-smile/TCL;)<br>
<br>

### 场景图生成(Scene Graph Generation)

**Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs(将视频场景图重新格式化为时间二分图)**<br>
*keywords: Video Scene Graph Generation, Transformer, Video Grounding*<br>
[paper](https://arxiv.org/abs/2112.04222) | [code](https://github.com/Dawn-LX/VidVRD-tracklets)<br>
<br>

<a name="MetricLearning"></a>
## 度量学习(Metric Learning)
<br>
        
### 场景图理解(Scene Graph Understanding)

**Weakly Supervised Object Localization as Domain Adaption(作为域适应的弱监督对象定位)**<br>
*keywords: Weakly Supervised Object Localization(WSOL), Multi-instance learning based WSOL, Separated-structure based WSOL, Domain Adaption*<br>
[paper](https://arxiv.org/abs/2203.01714) | [code](https://github.com/zh460045050/DA-WSOL_CVPR2022)<br>
<br>

<a name="ContrastiveLearning"></a>
## 对比学习(Contrastive Learning)
<br>
        
### 场景图理解(Scene Graph Understanding)

**Enhancing Adversarial Robustness for Deep Metric Learning(增强深度度量学习的对抗鲁棒性)**<br>
*keywords: Adversarial Attack, Adversarial Defense, Deep Metric Learning*<br>
[paper](https://arxiv.org/pdf/2203.01439.pdf)<br>
<br>
**HCSC: Hierarchical Contrastive Selective Coding(分层对比选择性编码)**<br>
*keywords: Self-supervised Representation Learning, Deep Clustering, Contrastive Learning*<br>
[paper](https://arxiv.org/abs/2202.00455) | [code](https://github.com/gyfastas/HCSC)<br>
<br>

<a name="Robotic"></a>
## 机器人(Robotic)
<br>
        
### 场景图理解(Scene Graph Understanding)

**Crafting Better Contrastive Views for Siamese Representation Learning(为连体表示学习制作更好的对比视图)**<br>
[paper](https://arxiv.org/pdf/2202.03278.pdf) | [code](https://github.com/xyupeng/ContrastiveCrop)<br>
<br>

<a name="self-supervisedlearning"></a>
## 自监督学习/半监督学习
<br>
        
### 场景图理解(Scene Graph Understanding)

**IFOR: Iterative Flow Minimization for Robotic Object Rearrangement(IFOR：机器人对象重排的迭代流最小化)**<br>
[paper](https://arxiv.org/pdf/2202.00732.pdf)<br>
<br>
**Class-Aware Contrastive Semi-Supervised Learning(类感知对比半监督学习)**<br>
*keywords: Semi-Supervised Learning, Self-Supervised Learning, Real-World Unlabeled Data Learning*<br>
[paper](https://arxiv.org/abs/2203.02261)<br>
<br>

<a name="None"></a>
## 暂无分类
<br>
        
### 场景图理解(Scene Graph Understanding)

**A study on the distribution of social biases in self-supervised learning visual models(自监督学习视觉模型中social biases分布的研究)**<br>
[paper](https://arxiv.org/pdf/2203.01854.pdf)<br>
<br>
**Do Explanations Explain? Model Knows Best(解释解释吗？ 模型最清楚)**<br>
[paper](https://arxiv.org/abs/2203.02269)<br>
<br>
**PINA: Learning a Personalized Implicit Neural Avatar from a Single RGB-D Video Sequence(PINA：从单个 RGB-D 视频序列中学习个性化的隐式神经化身)**<br>
[paper](https://arxiv.org/abs/2203.01754)<br>
<br>
