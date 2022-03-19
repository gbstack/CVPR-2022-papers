
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
        
### 2D目标检测(2D Object Detection)

**MUM : Mix Image Tiles and UnMix Feature Tiles for Semi-Supervised Object Detection(混合图像块和 UnMix 特征块用于半监督目标检测)**<br>
[paper](https://arxiv.org/abs/2111.10958) | [code](https://github.com/JongMokKim/mix-unmix)<br>
<br>
**SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection(域自适应对象检测的语义完全图匹配)**<br>
[paper](https://arxiv.org/abs/2203.06398) | [code](https://github.com/CityU-AIM-Group/SIGMA)<br>
<br>
**Accelerating DETR Convergence via Semantic-Aligned Matching(通过语义对齐匹配加速 DETR 收敛)**<br>
[paper](https://arxiv.org/abs/2203.06883) | [code](https://github.com/ZhangGongjie/SAM-DETR)<br>
<br>
**Focal and Global Knowledge Distillation for Detectors(探测器的焦点和全局知识蒸馏)**<br>
*keywords: Object Detection, Knowledge Distillation*<br>
[paper](https://arxiv.org/abs/2111.11837) | [code](https://github.com/yzd-v/FGD)<br>
<br>
**Unknown-Aware Object Detection: Learning What You Don't Know from Videos in the Wild(未知感知对象检测：从野外视频中学习你不知道的东西)**<br>
[paper](https://arxiv.org/abs/2203.03800) | [code](https://github.com/deeplearning-wisc/stud)<br>
<br>
**Localization Distillation for Dense Object Detection(密集对象检测的定位蒸馏)**<br>
*keywords: Bounding Box Regression, Localization Quality Estimation, Knowledge Distillation*<br>
[paper](https://arxiv.org/abs/2102.12252) | [code](https://github.com/HikariTJU/LD)<br>
<br>

### 视频目标检测(Video Object Detection)

**Unsupervised Activity Segmentation by Joint Representation Learning and Online Clustering(通过联合表示学习和在线聚类进行无监督活动分割)**<br>
[paper](https://arxiv.org/abs/2105.13353)<br>
<br>

### 3D目标检测(3D object detection)

**MonoJSG: Joint Semantic and Geometric Cost Volume for Monocular 3D Object Detection(单目 3D 目标检测的联合语义和几何成本量)**<br>
[paper](https://arxiv.org/abs/2203.08563) | [code](https://github.com/lianqing11/MonoJSG)<br>
<br>
**DeepFusion: Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection(用于多模态 3D 目标检测的激光雷达相机深度融合)**<br>
[paper](https://arxiv.org/abs/2203.08195) | [code](https://github.com/tensorflow/lingvo/tree/master/lingvo/)<br>
<br>
**Point Density-Aware Voxels for LiDAR 3D Object Detection(用于 LiDAR 3D 对象检测的点密度感知体素)**<br>
[paper](https://arxiv.org/abs/2203.05662) | [code](https://github.com/TRAILab/PDV)<br>
<br>
**Back to Reality: Weakly-supervised 3D Object Detection with Shape-guided Label Enhancement(带有形状引导标签增强的弱监督 3D 对象检测)**<br>
[paper](https://arxiv.org/abs/2203.05238) | [code](https://github.com/xuxw98/BackToReality)<br>
<br>
**Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes(在 3D 场景中实现稳健的定向边界框检测)**<br>
[paper](https://arxiv.org/abs/2011.12001) | [code](https://github.com/qq456cvb/CanonicalVoting)<br>
<br>
**A Versatile Multi-View Framework for LiDAR-based 3D Object Detection with Guidance from Panoptic Segmentation(在全景分割的指导下，用于基于 LiDAR 的 3D 对象检测的多功能多视图框架)**<br>
*keywords: 3D Object Detection with Point-based Methods, 3D Object Detection with Grid-based Methods, Cluster-free 3D Panoptic Segmentation, CenterPoint 3D Object Detection*<br>
[paper](https://arxiv.org/abs/2203.02133)<br>
<br>
**Pseudo-Stereo for Monocular 3D Object Detection in Autonomous Driving(自动驾驶中用于单目 3D 目标检测的伪立体)**<br>
*keywords: Autonomous Driving, Monocular 3D Object Detection*<br>
[paper](https://arxiv.org/abs/2203.02112) | [code](https://github.com/revisitq/Pseudo-Stereo-3D)<br>
<br>

### 伪装目标检测(Camouflaged Object Detection)

**Implicit Motion Handling for Video Camouflaged Object Detection(视频伪装对象检测的隐式运动处理)**<br>
[paper](https://arxiv.org/abs/2203.07363)<br>
<br>
**Zoom In and Out: A Mixed-scale Triplet Network for Camouflaged Object Detection(放大和缩小：用于伪装目标检测的混合尺度三元组网络)**<br>
[paper](https://arxiv.org/abs/2203.02688) | [code](https://github.com/lartpang/ZoomNet)<br>
<br>

### 显著性目标检测(Saliency Object Detection)

**Bi-directional Object-context Prioritization Learning for Saliency Ranking(显着性排名的双向对象上下文优先级学习)**<br>
[paper](https://arxiv.org/abs/2203.09416) | [code](https://github.com/GrassBro/OCOR)<br>
<br>
**Democracy Does Matter: Comprehensive Feature Mining for Co-Salient Object Detection()**<br>
[paper](https://arxiv.org/abs/2203.05787)<br>
<br>

### 关键点检测(Keypoint Detection)

**UKPGAN: A General Self-Supervised Keypoint Detector(一个通用的自监督关键点检测器)**<br>
[paper](https://arxiv.org/abs/2011.11974) | [code](https://github.com/qq456cvb/UKPGAN)<br>
<br>

### 车道线检测(Lane Detection)

**Rethinking Efficient Lane Detection via Curve Modeling(通过曲线建模重新思考高效车道检测)**<br>
*keywords: Segmentation-based Lane Detection, Point Detection-based Lane Detection, Curve-based Lane Detection, autonomous driving*<br>
[paper](https://arxiv.org/abs/2203.02431) | [code](https://github.com/voldemortX/pytorch-auto-drive)<br>
<br>

### 边缘检测(Edge Detection)

**EDTER: Edge Detection with Transformer(使用transformer的边缘检测)**<br>
[paper](https://arxiv.org/abs/2203.08566) | [code](https://github.com/MengyangPu/EDTER)<br>
<br>

### 消失点检测(Vanishing Point Detection)

**Deep vanishing point detection: Geometric priors make dataset variations vanish(深度**消失点检测**：几何先验使数据集变化消失)**<br>
[paper](https://arxiv.org/abs/2203.08586) | [code](https://github.com/yanconglin/VanishingPoint_HoughTransform_GaussianSphere)<br>
<br>

<a name="Segmentation"></a>
## 分割(Segmentation)
<br>
        
### 图像分割(Image Segmentation)

**Learning What Not to Segment: A New Perspective on Few-Shot Segmentation(学习不分割的内容：关于小样本分割的新视角)**<br>
[paper](https://arxiv.org/abs/2203.07615) | [code](http://github.com/chunbolang/BAM)<br>
<br>
**CRIS: CLIP-Driven Referring Image Segmentation(CLIP 驱动的参考图像分割)**<br>
[paper](https://arxiv.org/abs/2111.15174)<br>
<br>
**Hyperbolic Image Segmentation(双曲线图像分割)**<br>
[paper](https://arxiv.org/abs/2203.05898)<br>
<br>

### 全景分割(Panoptic Segmentation)

**Bending Reality: Distortion-aware Transformers for Adapting to Panoramic Semantic Segmentation(弯曲现实：适应全景语义分割的失真感知Transformer)**<br>
*keywords: Semanticand panoramic segmentation, Unsupervised domain adaptation, Transformer*<br>
[paper](https://arxiv.org/abs/2203.01452) | [code](https://github.com/jamycheung/Trans4PASS)<br>
<br>

### 语义分割(Semantic Segmentation)

**Scribble-Supervised LiDAR Semantic Segmentation**<br>
[paper](https://arxiv.org/abs/2203.08537) | [code](http://github.com/ouenal/scribblekitti)<br>
<br>
**ADAS: A Direct Adaptation Strategy for Multi-Target Domain Adaptive Semantic Segmentation(多目标域自适应语义分割的直接适应策略)**<br>
[paper](https://arxiv.org/abs/2203.06811)<br>
<br>
**Weakly Supervised Semantic Segmentation by Pixel-to-Prototype Contrast(通过像素到原型对比的弱监督语义分割)**<br>
[paper](https://arxiv.org/abs/2110.07110)<br>
<br>
**Representation Compensation Networks for Continual Semantic Segmentation(连续语义分割的表示补偿网络)**<br>
[paper](https://arxiv.org/abs/2203.05402) | [code](https://github.com/zhangchbin/RCIL)<br>
<br>
**Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels(使用不可靠伪标签的半监督语义分割)**<br>
[paper](https://arxiv.org/abs/2203.03884) | [code](https://github.com/Haochen-Wang409/U2PL/)<br>
<br>
**Weakly Supervised Semantic Segmentation using Out-of-Distribution Data(使用分布外数据的弱监督语义分割)**<br>
[paper](https://arxiv.org/abs/2203.03860) | [code](https://github.com/naver-ai/w-ood)<br>
<br>
**Self-supervised Image-specific Prototype Exploration for Weakly Supervised Semantic Segmentation(弱监督语义分割的自监督图像特定原型探索)**<br>
[paper](https://arxiv.org/abs/2203.02909) | [code](https://github.com/chenqi1126/SIPE)<br>
<br>
**Multi-class Token Transformer for Weakly Supervised Semantic Segmentation(用于弱监督语义分割的多类token Transformer)**<br>
[paper](https://arxiv.org/abs/2203.02891) | [code](https://github.com/xulianuwa/MCTformer)<br>
<br>
**Cross Language Image Matching for Weakly Supervised Semantic Segmentation(用于弱监督语义分割的跨语言图像匹配)**<br>
[paper](https://arxiv.org/abs/2203.02668)<br>
<br>
**Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers(从注意力中学习亲和力：使用 Transformers 的端到端弱监督语义分割)**<br>
[paper](https://arxiv.org/abs/2203.02664) | [code](https://github.com/rulixiang/afa)<br>
<br>
**ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation(让自我训练更好地用于半监督语义分割)**<br>
*keywords: Semi-supervised learning, Semantic segmentation, Uncertainty estimation*<br>
[paper](https://arxiv.org/abs/2106.05095) | [code](https://github.com/LiheYoung/ST-PlusPlus)<br>
<br>
**Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation(弱监督语义分割的类重新激活图)**<br>
[paper](https://arxiv.org/pdf/2203.00962.pdf) | [code](https://github.com/zhaozhengChen/ReCAM)<br>
<br>

### 实例分割(Instance Segmentation)

**E2EC: An End-to-End Contour-based Method for High-Quality High-Speed Instance Segmentation(一种基于端到端轮廓的高质量高速实例分割方法)**<br>
[paper](https://arxiv.org/abs/2203.04074) | [code](https://github.com/zhang-tao-whu/e2ec)<br>
<br>
**Efficient Video Instance Segmentation via Tracklet Query and Proposal(通过 Tracklet Query 和 Proposal 进行高效的视频实例分割)**<br>
[paper](https://arxiv.org/abs/2203.01853)<br>
<br>
**SoftGroup for 3D Instance Segmentation on Point Clouds(用于点云上的 3D 实例分割)**<br>
*keywords: 3D Vision, Point Clouds, Instance Segmentation*<br>
[paper](https://arxiv.org/abs/2203.01509) | [code](https://github.com/thangvubk/SoftGroup.git)<br>
<br>

### 视频目标分割(Video Object Segmentation)

**Language as Queries for Referring Video Object Segmentation(语言作为引用视频对象分割的查询)**<br>
[paper](https://arxiv.org/abs/2201.00487) | [code](https://github.com/wjn922/ReferFormer)<br>
<br>

<a name="Estimation"></a>
## 估计(Estimation)
<br>
        
### 光流/运动估计(Optical Flow/Motion Estimation)

**CamLiFlow: Bidirectional Camera-LiDAR Fusion for Joint Optical Flow and Scene Flow Estimation(用于联合光流和场景流估计的双向相机-LiDAR 融合)**<br>
[paper](https://arxiv.org/abs/2111.10502)<br>
<br>

### 深度估计(Depth Estimation)

**ChiTransformer:Towards Reliable Stereo from Cues(从线索走向可靠的立体声)**<br>
[paper](https://arxiv.org/abs/2203.04554)<br>
<br>
**Rethinking Depth Estimation for Multi-View Stereo: A Unified Representation and Focal Loss(重新思考多视图立体的深度估计：统一表示和焦点损失)**<br>
[paper](https://arxiv.org/abs/2201.01501) | [code](https://github.com/prstrive/UniMVSNet)<br>
<br>
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
**OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion(通过几何感知融合进行 360 度单目深度估计)**<br>
*keywords: monocular depth estimation(单目深度估计),transformer*<br>
[paper](https://arxiv.org/abs/2203.00838)<br>
<br>

### 人体解析/人体姿态估计(Human Parsing/Human Pose Estimation)

**Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation from Monocular Video(捕捉运动中的人类：来自单目视频的时间注意 3D 人体姿势和形状估计)**<br>
[paper](https://arxiv.org/abs/2203.08534)<br>
<br>
**Physical Inertial Poser (PIP): Physics-aware Real-time Human Motion Tracking from Sparse Inertial Sensors(来自稀疏惯性传感器的物理感知实时人体运动跟踪)**<br>
[paper](https://arxiv.org/abs/2203.08528)<br>
<br>
**Distribution-Aware Single-Stage Models for Multi-Person 3D Pose Estimation(用于多人 3D 姿势估计的分布感知单阶段模型)**<br>
[paper](https://arxiv.org/abs/2203.07697)<br>
<br>
**MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation(用于 3D 人体姿势估计的多假设transformer)**<br>
[paper](https://arxiv.org/abs/2111.12707) | [code](https://github.com/Vegetebird/MHFormer)<br>
<br>
**CDGNet: Class Distribution Guided Network for Human Parsing(用于人类解析的类分布引导网络)**<br>
[paper](https://arxiv.org/abs/2111.14173)<br>
<br>
**Forecasting Characteristic 3D Poses of Human Actions(预测人类行为的特征 3D 姿势)**<br>
[paper](https://arxiv.org/abs/2011.15079)<br>
<br>
**Learning Local-Global Contextual Adaptation for Multi-Person Pose Estimation(学习用于多人姿势估计的局部-全局上下文适应)**<br>
*keywords: Top-Down Pose Estimation(从上至下姿态估计), Limb-based Grouping, Direct Regression*<br>
[paper](https://arxiv.org/pdf/2109.03622.pdf)<br>
<br>
**MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video(用于视频中 3D 人体姿势估计的 Seq2seq 混合时空编码器)**<br>
[paper](https://arxiv.org/pdf/2203.00859.pdf)<br>
<br>

<a name="ImageProcessing"></a>
## 图像处理(Image Processing)
<br>
        
### 超分辨率(Super Resolution)

**A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution(一种用于空间变形鲁棒场景文本图像超分辨率的文本注意网络)**<br>
[paper](https://arxiv.org/abs/2203.09388) | [code](https://github.com/mjq11302010044/TATT)<br>
<br>
**Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution(一种真实图像超分辨率的局部判别学习方法)**<br>
[paper](https://arxiv.org/abs/2203.09195) | [code](https://github.com/csjliang/LDL)<br>
<br>
**Blind Image Super-resolution with Elaborate Degradation Modeling on Noise and Kernel(对噪声和核进行精细退化建模的盲图像超分辨率)**<br>
[paper](https://arxiv.org/abs/2107.00986) | [code](https://github.com/zsyOAOA/BSRDM)<br>
<br>
**Reflash Dropout in Image Super-Resolution(图像超分辨率中的闪退dropout)**<br>
[paper](https://arxiv.org/abs/2112.12089)<br>
<br>
**Towards Bidirectional Arbitrary Image Rescaling: Joint Optimization and Cycle Idempotence(迈向双向任意图像缩放：联合优化和循环幂等)**<br>
[paper](https://arxiv.org/abs/2203.00911)<br>
<br>
**HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening(用于全色锐化的纹理和光谱特征融合Transformer)**<br>
[paper](https://arxiv.org/abs/2203.02503) | [code](https://github.com/wgcban/HyperTransformer)<br>
<br>
**HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging(光谱压缩成像的高分辨率双域学习)**<br>
*keywords: HSI Reconstruction, Self-Attention Mechanism,  Image Frequency Spectrum Analysis*<br>
[paper](https://arxiv.org/pdf/2203.02149.pdf)<br>
<br>

### 图像复原/图像增强/图像重建(Image Restoration/Image Reconstruction)

**Restormer: Efficient Transformer for High-Resolution Image Restoration(用于高分辨率图像复原的高效transformer)**<br>
[paper](https://arxiv.org/abs/2111.09881) | [code](https://github.com/swz30/Restormer)<br>
<br>
**Event-based Video Reconstruction via Potential-assisted Spiking Neural Network(通过电位辅助尖峰神经网络进行基于事件的视频重建)**<br>
[paper](https://arxiv.org/pdf/2201.10943.pdf)<br>
<br>

### 图像去噪/去模糊/去雨去雾(Image Denoising)

**Neural Compression-Based Feature Learning for Video Restoration(用于视频复原的基于神经压缩的特征学习)(**视频处理**)**<br>
[paper](https://arxiv.org/abs/2203.09208)<br>
<br>
**Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots(具有可见盲点的自监督图像去噪)**<br>
[paper](https://arxiv.org/abs/2203.06967) | [code](https://github.com/demonsjin/Blind2Unblind)<br>
<br>
**E-CIR: Event-Enhanced Continuous Intensity Recovery(事件增强的连续强度恢复)**<br>
*keywords: Event-Enhanced Deblurring, Video Representation*<br>
[paper](https://arxiv.org/abs/2203.01935) | [code](https://github.com/chensong1995/E-CIR)<br>
<br>

### 图像编辑/图像修复(Image Edit/Inpainting)

**High-Fidelity GAN Inversion for Image Attribute Editing(用于图像属性编辑的高保真 GAN 反演)**<br>
[paper](https://arxiv.org/abs/2109.06590) | [code](https://github.com/Tengfei-Wang/HFGI)<br>
<br>
**Style Transformer for Image Inversion and Editing(用于图像反转和编辑的样式transformer)**<br>
[paper](https://arxiv.org/abs/2203.07932) | [code](https://github.com/sapphire497/style-transformer)<br>
<br>
**MISF: Multi-level Interactive Siamese Filtering for High-Fidelity Image Inpainting(用于高保真图像修复的多级交互式 Siamese 过滤)**<br>
[paper](https://arxiv.org/abs/2203.06304) | [code](https://github.com/tsingqguo/misf)<br>
<br>
**HairCLIP: Design Your Hair by Text and Reference Image(通过文本和参考图像设计你的头发)**<br>
*keywords: Language-Image Pre-Training (CLIP), Generative Adversarial Networks*<br>
[paper](https://arxiv.org/abs/2112.05142)<br>
<br>
**Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding(增量transformer结构增强图像修复与掩蔽位置编码)**<br>
*keywords: Image Inpainting, Transformer, Image Generation*<br>
[paper](https://arxiv.org/abs/2203.00867) | [code](https://github.com/DQiaole/ZITS_inpainting)<br>
<br>

### 图像翻译(Image Translation)

**QS-Attn: Query-Selected Attention for Contrastive Learning in I2I Translation(图像翻译中对比学习的查询选择注意)**<br>
[paper](https://arxiv.org/abs/2203.08483) | [code](https://github.com/sapphire497/query-selected-attention)<br>
<br>
**FlexIT: Towards Flexible Semantic Image Translation(迈向灵活的语义图像翻译)**<br>
[paper](https://arxiv.org/abs/2203.04705)<br>
<br>
**Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks(探索图像到图像翻译任务中对比学习的补丁语义关系)**<br>
*keywords: image translation, knowledge transfer,Contrastive learning*<br>
[paper](https://arxiv.org/pdf/2203.01532.pdf)<br>
<br>

### 风格迁移(Style Transfer)

**Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization(任意风格迁移和域泛化的精确特征分布匹配)**<br>
[paper](https://arxiv.org/abs/2203.07740) | [code](https://github.com/YBZh/EFDM)<br>
<br>
**Style-ERD: Responsive and Coherent Online Motion Style Transfer(响应式和连贯的在线运动风格迁移)**<br>
[paper](https://arxiv.org/abs/2203.02574)<br>
<br>
**CLIPstyler: Image Style Transfer with a Single Text Condition(具有单一文本条件的图像风格转移)**<br>
*keywords: Style Transfer, Text-guided synthesis, Language-Image Pre-Training (CLIP)*<br>
[paper](https://arxiv.org/abs/2112.00374)<br>
<br>

<a name="Face"></a>
## 人脸(Face)
<br>
        
### 人脸(Face)

**FaceFormer: Speech-Driven 3D Facial Animation with Transformers(FaceFormer：带有transformer的语音驱动的 3D 面部动画)**<br>
[paper](https://arxiv.org/abs/2112.05329) | [code](https://evelynfan.github.io/audio2face/)<br>
<br>
**Sparse Local Patch Transformer for Robust Face Alignment and Landmarks Inherent Relation Learning(用于鲁棒人脸对齐和地标固有关系学习的稀疏局部补丁transformer)**<br>
[paper](https://arxiv.org/abs/2203.06541) | [code](https://github.com/Jiahao-UTS/SLPT-master)<br>
<br>

### 人脸识别/检测(Facial Recognition/Detection)

**Privacy-preserving Online AutoML for Domain-Specific Face Detection(用于特定领域人脸检测的隐私保护在线 AutoML)**<br>
[paper](https://arxiv.org/abs/2203.08399)<br>
<br>
**An Efficient Training Approach for Very Large Scale Face Recognition(一种有效的超大规模人脸识别训练方法)**<br>
[paper](https://arxiv.org/pdf/2105.10375.pdf) | [code](https://github.com/tiandunx/FFC)<br>
<br>

### 人脸生成/合成/重建/编辑(Face Generation/Face Synthesis/Face Reconstruction/Face Editing)

**GCFSR: a Generative and Controllable Face Super Resolution Method Without Facial and GAN Priors(一种没有面部和 GAN 先验的生成可控人脸超分辨率方法)**<br>
[paper](https://arxiv.org/abs/2203.07319)<br>
<br>
**Sparse to Dense Dynamic 3D Facial Expression Generation(稀疏到密集的动态 3D 面部表情生成)**<br>
*keywords: Facial expression generation, 4D face generation, 3D face modeling*<br>
[paper](https://arxiv.org/pdf/2105.07463.pdf)<br>
<br>

### 人脸伪造/反欺骗(Face Forgery/Face Anti-Spoofing)

**Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing(通过 Shuffled Style Assembly 进行域泛化以进行人脸反欺骗)**<br>
[paper](https://arxiv.org/abs/2203.05340) | [code](https://github.com/wangzhuo2019/SSAN)<br>
<br>
**Voice-Face Homogeneity Tells Deepfake**<br>
[paper](https://arxiv.org/abs/2203.02195) | [code](https://github.com/xaCheng1996/VFD)<br>
<br>
**Protecting Celebrities with Identity Consistency Transformer(使用身份一致性transformer保护名人)**<br>
[paper](https://arxiv.org/abs/2203.01318)<br>
<br>

<a name="ObjectTracking"></a>
## 目标跟踪(Object Tracking)
<br>
        
### 目标跟踪(Object Tracking)

**Iterative Corresponding Geometry: Fusing Region and Depth for Highly Efficient 3D Tracking of Textureless Objects(迭代对应几何：融合区域和深度以实现无纹理对象的高效 3D 跟踪)**<br>
[paper](https://arxiv.org/abs/2203.05334) | [code](https://github.com/DLR- RM/3DObjectTracking)<br>
<br>
**TCTrack: Temporal Contexts for Aerial Tracking(空中跟踪的时间上下文)**<br>
[paper](https://arxiv.org/abs/2203.01885) | [code](https://github.com/vision4robotics/TCTrack)<br>
<br>
**Beyond 3D Siamese Tracking: A Motion-Centric Paradigm for 3D Single Object Tracking in Point Clouds(超越 3D 连体跟踪：点云中 3D 单对象跟踪的以运动为中心的范式)**<br>
*keywords: Single Object Tracking, 3D Multi-object Tracking / Detection, Spatial-temporal Learning on Point Clouds*<br>
[paper](https://arxiv.org/abs/2203.01730)<br>
<br>
**Correlation-Aware Deep Tracking(相关感知深度跟踪)**<br>
[paper](https://arxiv.org/abs/2203.01666)<br>
<br>

<a name="ImageRetrieval"></a>
## 图像&视频检索/视频理解(Image&Video Retrieval/Video Understanding)
<br>
        
### 图像&视频检索/视频理解(Image&Video Retrieval/Video Understanding)

**Bridging Video-text Retrieval with Multiple Choice Questions(桥接视频文本检索与多项选择题)**<br>
[paper](https://arxiv.org/abs/2201.04850) | [code](https://github.com/TencentARC/MCQ)<br>
<br>
**BEVT: BERT Pretraining of Video Transformers(视频Transformer的 BERT 预训练)**<br>
*keywords: Video understanding, Vision transformers, Self-supervised representation learning, BERT pretraining*<br>
[paper](https://arxiv.org/abs/2112.01529) | [code](https://github.com/xyzforever/BEVT)<br>
<br>

### 行为识别/动作识别/检测/分割/定位(Action/Activity Recognition)

**Spatio-temporal Relation Modeling for Few-shot Action Recognition(小样本动作识别的时空关系建模)**<br>
[paper](https://arxiv.org/abs/2112.05132) | [code](https://github.com/Anirudh257/strm)<br>
<br>
**RCL: Recurrent Continuous Localization for Temporal Action Detection(用于时间动作检测的循环连续定位)**<br>
[paper](https://arxiv.org/abs/2203.07112)<br>
<br>
**OpenTAL: Towards Open Set Temporal Action Localization(走向开放集时间动作定位)**<br>
[paper](https://arxiv.org/abs/2203.05114) | [code](https://www.rit.edu/actionlab/opental)<br>
<br>
**End-to-End Semi-Supervised Learning for Video Action Detection(视频动作检测的端到端半监督学习)**<br>
[paper](https://arxiv.org/abs/2203.04251)<br>
<br>
**Learnable Irrelevant Modality Dropout for Multimodal Action Recognition on Modality-Specific Annotated Videos(模态特定注释视频上多模态动作识别的可学习不相关模态丢失)**<br>
[paper](https://arxiv.org/abs/2203.03014)<br>
<br>
**Weakly Supervised Temporal Action Localization via Representative Snippet Knowledge Propagation(通过代表性片段知识传播的弱监督时间动作定位)**<br>
[paper](https://arxiv.org/abs/2203.02925) | [code](https://github.com/LeonHLJ/RSKP)<br>
<br>
**Colar: Effective and Efficient Online Action Detection by Consulting Exemplars(通过咨询示例进行有效且高效的在线动作检测)**<br>
*keywords: Online action detection(在线动作检测)*<br>
[paper](https://arxiv.org/pdf/2203.01057.pdf)<br>
<br>

### 图像/视频字幕(Image/Video Caption)

**Hierarchical Modular Network for Video Captioning(用于视频字幕的分层模块化网络)**<br>
[paper](https://arxiv.org/abs/2111.12476) | [code](https://github.com/MarcusNerva/HMN)<br>
<br>
**X -Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning(使用 Transformer 进行 3D 密集字幕的跨模式知识迁移)**<br>
[paper](https://arxiv.org/pdf/2203.00843.pdf)<br>
<br>

<a name="MedicalImaging"></a>
## 医学影像(Medical Imaging)
<br>
        
### 医学影像(Medical Imaging)

**Vox2Cortex: Fast Explicit Reconstruction of Cortical Surfaces from 3D MRI Scans with Geometric Deep Neural Networks(使用几何深度神经网络从 3D MRI 扫描中快速显式重建皮质表面)**<br>
[paper](https://arxiv.org/abs/2203.09446) | [code](https://github.com/ai-med/Vox2Cortex)<br>
<br>
**Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization(通过风格增强和双重归一化的可泛化跨模态医学图像分割)**<br>
[paper](https://arxiv.org/abs/2112.11177) | [code](https://github.com/zzzqzhou/Dual-Normalization)<br>
<br>
**Adaptive Early-Learning Correction for Segmentation from Noisy Annotations(从噪声标签中分割的自适应早期学习校正)**<br>
*keywords: medical-imaging segmentation, Noisy Annotations*<br>
[paper](https://arxiv.org/abs/2110.03740) | [code](https://github.com/Kangningthu/ADELE)<br>
<br>
**Temporal Context Matters: Enhancing Single Image Prediction with Disease Progression Representations(时间上下文很重要：使用疾病进展表示增强单图像预测)**<br>
*keywords: Self-supervised Transformer, Temporal modeling of disease progression*<br>
[paper](https://arxiv.org/abs/2203.01933)<br>
<br>

<a name="TDR"></a>
## 文本检测/识别/理解(Text Detection/Recognition/Understanding)
<br>
        
### 文本检测/识别/理解(Text Detection/Recognition/Understanding)

**XYLayoutLM: Towards Layout-Aware Multimodal Networks For Visually-Rich Document Understanding(迈向布局感知多模式网络，以实现视觉丰富的文档理解)**<br>
[paper](https://arxiv.org/abs/2203.06947)<br>
<br>

<a name="GAN"></a>
## GAN/生成式/对抗式(GAN/Generative/Adversarial)
<br>
        
### GAN/生成式/对抗式(GAN/Generative/Adversarial)

**Improving the Transferability of Targeted Adversarial Examples through Object-Based Diverse Input(通过基于对象的多样化输入提高目标对抗样本的可迁移性)**<br>
[paper](https://arxiv.org/abs/2203.09123) | [code](https://github.com/dreamflake/ODI)<br>
<br>
**Towards Practical Certifiable Patch Defense with Vision Transformer(使用 Vision Transformer 实现实用的可认证补丁防御)**<br>
[paper](https://arxiv.org/abs/2203.08519)<br>
<br>
**Few Shot Generative Model Adaption via Relaxed Spatial Structural Alignment(基于松弛空间结构对齐的小样本生成模型自适应)**<br>
[paper](https://arxiv.org/abs/2203.04121)<br>
<br>
**Enhancing Adversarial Training with Second-Order Statistics of Weights(使用权重的二阶统计加强对抗训练)**<br>
[paper](https://arxiv.org/abs/2203.06020) | [code](https://github.com/Alexkael/S2O)<br>
<br>
**Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack(通过自适应自动攻击对对抗鲁棒性的实际评估)**<br>
[paper](https://arxiv.org/abs/2203.05154)<br>
<br>
**Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity(对语义相似性的频率驱动的不可察觉的对抗性攻击)**<br>
[paper](https://arxiv.org/abs/2203.05151)<br>
<br>
**Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon(阴影可能很危险：自然现象的隐秘而有效的物理世界对抗性攻击)**<br>
[paper](https://arxiv.org/abs/2203.03818)<br>
<br>
**Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer(保护面部隐私：通过风格稳健的化妆转移生成对抗性身份面具)**<br>
[paper](https://arxiv.org/pdf/2203.03121.pdf)<br>
<br>
**Adversarial Texture for Fooling Person Detectors in the Physical World(物理世界中愚弄人探测器的对抗性纹理)**<br>
[paper](https://arxiv.org/abs/2203.03373)<br>
<br>
**Label-Only Model Inversion Attacks via Boundary Repulsion(通过边界排斥的仅标签模型反转攻击)**<br>
[paper](https://arxiv.org/pdf/2203.01925.pdf)<br>
<br>

<a name="IGIS"></a>
## 图像生成/图像合成/视频生成(Image Generation/Image Synthesis/Video Generation)
<br>
        
### 图像生成/图像合成/视频生成(Image Generation/Image Synthesis/Video Generation)

**Modulated Contrast for Versatile Image Synthesis(用于多功能图像合成的调制对比度)**<br>
[paper](https://arxiv.org/abs/2203.09333) | [code](https://github.com/fnzhan/MoNCE)<br>
<br>
**Attribute Group Editing for Reliable Few-shot Image Generation(属性组编辑用于可靠的小样本图像生成)**<br>
[paper](https://arxiv.org/abs/2203.08422) | [code](https://github.com/UniBester/AGE)<br>
<br>
**Text to Image Generation with Semantic-Spatial Aware GAN(使用语义空间感知 GAN 生成文本到图像)**<br>
[paper](https://arxiv.org/abs/2104.00567) | [code](https://github.com/wtliao/text2image)<br>
<br>
**Playable Environments: Video Manipulation in Space and Time(可播放环境：空间和时间的视频操作)**<br>
[paper](https://arxiv.org/abs/2203.01914) | [code](https://willi-menapace.github.io/playable-environments-website)<br>
<br>
**Depth-Aware Generative Adversarial Network for Talking Head Video Generation(用于说话头视频生成的深度感知生成对抗网络)**<br>
[paper](https://arxiv.org/abs/2203.06605) | [code](https://github.com/harlanhong/CVPR2022-DaGAN)<br>
<br>
**FLAG: Flow-based 3D Avatar Generation from Sparse Observations(从稀疏观察中生成基于流的 3D 头像)**<br>
[paper](https://arxiv.org/abs/2203.05789)<br>
<br>
**Dynamic Dual-Output Diffusion Models(动态双输出扩散模型)**<br>
[paper](https://arxiv.org/abs/2203.04304)<br>
<br>
**Exploring Dual-task Correlation for Pose Guided Person Image Generation(探索姿势引导人物图像生成的双任务相关性)**<br>
[paper](https://arxiv.org/abs/2203.02910) | [code](https://github.com/PangzeCheung/Dual-task-Pose-Transformer-Network)<br>
<br>
**Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning(告诉我什么并告诉我如何：通过多模式调节进行视频合成)**<br>
[paper](https://arxiv.org/abs/2203.02573) | [code](https://github.com/snap-research/MMVID)<br>
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
        
### 三维视觉(3D Vision)

**X -Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning(使用 Transformer 进行 3D 密集字幕的跨模式知识迁移)**<br>
[paper](https://arxiv.org/pdf/2203.00843.pdf)<br>
<br>

### 点云(Point Cloud)

**AutoGPart: Intermediate Supervision Search for Generalizable 3D Part Segmentation(通用 3D 零件分割的中间监督搜索)**<br>
[paper](https://arxiv.org/abs/2203.06558)<br>
<br>
**Geometric Transformer for Fast and Robust Point Cloud Registration(用于快速和稳健点云配准的几何transformer)**<br>
[paper](https://arxiv.org/abs/2202.06688) | [code](https://github.com/qinzheng93/GeoTransformer)<br>
<br>
**Contrastive Boundary Learning for Point Cloud Segmentation(点云分割的对比边界学习)**<br>
[paper](https://arxiv.org/abs/2203.05272) | [code](https://github.com/LiyaoTang/contrastBoundary)<br>
<br>
**Shape-invariant 3D Adversarial Point Clouds(形状不变的 3D 对抗点云)**<br>
[paper](https://arxiv.org/abs/2203.04041) | [code](https://github.com/shikiw/SI-Adv)<br>
<br>
**ART-Point: Improving Rotation Robustness of Point Cloud Classifiers via Adversarial Rotation(通过对抗旋转提高点云分类器的旋转鲁棒性)**<br>
[paper](https://arxiv.org/abs/2203.03888)<br>
<br>
**Lepard: Learning partial point cloud matching in rigid and deformable scenes(Lepard：在刚性和可变形场景中学习部分点云匹配)**<br>
[paper](https://arxiv.org/abs/2111.12591) | [code](https://github.com/rabbityl/lepard)<br>
<br>
**A Unified Query-based Paradigm for Point Cloud Understanding(一种基于统一查询的点云理解范式)**<br>
[paper](https://arxiv.org/pdf/2203.01252.pdf)<br>
<br>
**CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding(用于 3D 点云理解的自监督跨模态对比学习)**<br>
*keywords: Self-Supervised Learning, Contrastive Learning, 3D Point Cloud, Representation Learning, Cross-Modal Learning*<br>
[paper](https://arxiv.org/abs/2203.00680) | [code](http://github.com/MohamedAfham/CrossPoint)<br>
<br>

### 三维重建(3D Reconstruction)

**AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation(用于 3D 完成、重建和生成的形状先验)**<br>
[paper](https://arxiv.org/abs/2203.09516)<br>
<br>
**Interacting Attention Graph for Single Image Two-Hand Reconstruction(单幅图像双手重建的交互注意力图)**<br>
[paper](https://arxiv.org/abs/2203.09364) | [code](https://github.com/Dw1010/IntagHand)<br>
<br>
**OcclusionFusion: Occlusion-aware Motion Estimation for Real-time Dynamic 3D Reconstruction(实时动态 3D 重建的遮挡感知运动估计)**<br>
[paper](https://arxiv.org/abs/2203.07977)<br>
<br>
**Neural RGB-D Surface Reconstruction(神经 RGB-D 表面重建)**<br>
[paper](https://arxiv.org/abs/2104.04532)<br>
<br>
**Neural Face Identification in a 2D Wireframe Projection of a Manifold Object(流形对象的二维线框投影中的神经人脸识别)**<br>
[paper](https://arxiv.org/abs/2203.04229) | [code](https://manycore- research.github.io/faceformer)<br>
<br>
**Generating 3D Bio-Printable Patches Using Wound Segmentation and Reconstruction to Treat Diabetic Foot Ulcers(使用伤口分割和重建生成 3D 生物可打印贴片以治疗糖尿病足溃疡)**<br>
*keywords: semantic segmentation, 3D reconstruction, 3D bio-printers*<br>
[paper](https://arxiv.org/pdf/2203.03814.pdf)<br>
<br>
**H4D: Human 4D Modeling by Learning Neural Compositional Representation(通过学习神经组合表示进行人体 4D 建模)**<br>
*keywords: 4D Representation(4D 表征),Human Body Estimation(人体姿态估计),Fine-grained Human Reconstruction(细粒度人体重建)*<br>
[paper](https://arxiv.org/pdf/2203.01247.pdf)<br>
<br>

### 场景重建/视图合成/新视角合成(Novel View Synthesis)

**StyleMesh: Style Transfer for Indoor 3D Scene Reconstructions(室内 3D 场景重建的风格转换)**<br>
[paper](https://arxiv.org/abs/2112.01530) | [code](https://github.com/lukasHoel/stylemesh)<br>
<br>
**Look Outside the Room: Synthesizing A Consistent Long-Term 3D Scene Video from A Single Image(向外看：从单个图像合成一致的长期 3D 场景视频)**<br>
[paper](https://arxiv.org/abs/2203.09457) | [code](https://github.com/xrenaa/Look-Outside-Room)<br>
<br>
**Point-NeRF: Point-based Neural Radiance Fields(基于点的神经辐射场)**<br>
[paper](https://arxiv.org/abs/2201.08845) | [code](https://github.com/Xharlie/pointnerf)<br>
<br>
**CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields(文本和图像驱动的神经辐射场操作)**<br>
*keywords: NeRF,  Image Generation and Manipulation, Language-Image Pre-Training (CLIP)*<br>
[paper](https://arxiv.org/abs/2112.05139) | [code](https://cassiepython.github.io/clipnerf/)<br>
<br>
**Point-NeRF: Point-based Neural Radiance Fields(基于点的神经辐射场)**<br>
[paper](https://arxiv.org/pdf/2201.08845.pdf) | [code](https://github.com/Xharlie/pointnerf)<br>
<br>

<a name="ModelCompression"></a>
## 模型压缩(Model Compression)
<br>
        
### 知识蒸馏(Knowledge Distillation)

**Decoupled Knowledge Distillation(解耦知识蒸馏)**<br>
[paper](https://arxiv.org/abs/2203.08679) | [code](https://github.com/megvii-research/mdistiller)<br>
<br>
**Wavelet Knowledge Distillation: Towards Efficient Image-to-Image Translation(小波知识蒸馏：迈向高效的图像到图像转换)**<br>
[paper](https://arxiv.org/abs/2203.06321)<br>
<br>
**Knowledge Distillation as Efficient Pre-training: Faster Convergence, Higher Data-efficiency, and Better Transferability(知识蒸馏作为高效的预训练：更快的收敛、更高的数据效率和更好的可迁移性)**<br>
[paper](https://arxiv.org/abs/2203.05180) | [code](https://github.com/CVMI-Lab/KDEP)<br>
<br>
**Focal and Global Knowledge Distillation for Detectors(探测器的焦点和全局知识蒸馏)**<br>
*keywords: Object Detection, Knowledge Distillation*<br>
[paper](https://arxiv.org/abs/2111.11837) | [code](https://github.com/yzd-v/FGD)<br>
<br>

### 剪枝(Pruning)

**Interspace Pruning: Using Adaptive Filter Representations to Improve Training of Sparse CNNs(空间剪枝：使用自适应滤波器表示来改进稀疏 CNN 的训练)**<br>
[paper](https://arxiv.org/abs/2203.07808)<br>
<br>

### 量化(Quantization)

**Implicit Feature Decoupling with Depthwise Quantization(使用深度量化的隐式特征解耦)**<br>
[paper](https://arxiv.org/abs/2203.08080)<br>
<br>
**IntraQ: Learning Synthetic Images with Intra-Class Heterogeneity for Zero-Shot Network Quantization(学习具有类内异质性的合成图像以进行零样本网络量化)**<br>
[paper](https://arxiv.org/abs/2111.09136) | [code](https://github.com/zysxmu/IntraQ)<br>
<br>

<a name="NNS"></a>
## 神经网络结构设计(Neural Network Structure Design)
<br>
        
### 神经网络结构设计(Neural Network Structure Design)

**BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning(学习探索样本关系以进行鲁棒表征学习)**<br>
*keywords: sample relationship, data scarcity learning, Contrastive Self-Supervised Learning, long-tailed recognition, zero-shot learning, domain generalization, self-supervised learning*<br>
[paper](https://arxiv.org/abs/2203.01522) | [code](https://github.com/zhihou7/BatchFormer)<br>
<br>

### CNN

**On the Integration of Self-Attention and Convolution(自注意力和卷积的整合)**<br>
[paper](https://arxiv.org/abs/2111.14556)<br>
<br>
**Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs(将内核扩展到 31x31：重新审视 CNN 中的大型内核设计)**<br>
[paper](https://arxiv.org/abs/2203.06717) | [code](https://github.com/megvii-research/RepLKNet)<br>
<br>
**DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos(视频中稀疏帧差异的端到端 CNN 推断)**<br>
*keywords: sparse convolutional neural network, video inference accelerating*<br>
[paper](https://arxiv.org/abs/2203.03996)<br>
<br>
**A ConvNet for the 2020s**<br>
[paper](https://arxiv.org/abs/2201.03545) | [code](https://github.com/facebookresearch/ConvNeXt)<br>
<br>

### Transformer

**Attribute Surrogates Learning and Spectral Tokens Pooling in Transformers for Few-shot Learning**<br>
[paper](https://arxiv.org/abs/2203.09064) | [code](https://github.com/StomachCold/HCTransformers)<br>
<br>
**NomMer: Nominate Synergistic Context in Vision Transformer for Visual Recognition(在视觉transformer中为视觉识别指定协同上下文)**<br>
[paper](https://arxiv.org/abs/2111.12994) | [code](https://github.com/TencentYoutuResearch/VisualRecognition-NomMer)<br>
<br>
**Delving Deep into the Generalization of Vision Transformers under Distribution Shifts(深入研究分布变化下的视觉Transformer的泛化)**<br>
*keywords: out-of-distribution (OOD) generalization, Vision Transformers*<br>
[paper](https://arxiv.org/abs/2106.07617) | [code](https://github.com/Phoenix1153/ViT_OOD_generalization)<br>
<br>
**Mobile-Former: Bridging MobileNet and Transformer(连接 MobileNet 和 Transformer)**<br>
*keywords: Light-weight convolutional neural networks(轻量卷积神经网络),Combination of CNN and ViT*<br>
[paper](https://arxiv.org/abs/2108.05895)<br>
<br>

### 神经网络架构搜索(NAS)

**Global Convergence of MAML and Theory-Inspired Neural Architecture Search for Few-Shot Learning(MAML 的全局收敛和受理论启发的神经架构搜索以进行 Few-Shot 学习)**<br>
[paper](https://arxiv.org/abs/2203.09137) | [code](https://github.com/YiteWang/MetaNTK-NAS)<br>
<br>
**β-DARTS: Beta-Decay Regularization for Differentiable Architecture Search(可微架构搜索的 Beta-Decay 正则化)**<br>
[paper](https://arxiv.org/abs/2203.01665)<br>
<br>

### MLP

**Dynamic MLP for Fine-Grained Image Classification by Leveraging Geographical and Temporal Information(利用地理和时间信息进行细粒度图像分类的动态 MLP)**<br>
[paper](https://arxiv.org/abs/2203.03253) | [code](https://github.com/ylingfeng/DynamicMLP.git)<br>
<br>
**Revisiting the Transferability of Supervised Pretraining: an MLP Perspective(重新审视监督预训练的可迁移性：MLP 视角)**<br>
[paper](https://arxiv.org/abs/2112.00496)<br>
<br>
**An Image Patch is a Wave: Quantum Inspired Vision MLP(图像补丁是波浪：量子启发的视觉 MLP)**<br>
[paper](https://arxiv.org/abs/2111.12294) | [code](https://github.com/huawei-noah/CV-Backbones/tree/master/wavemlp_pytorch)<br>
<br>

<a name="DataProcessing"></a>
## 数据处理(Data Processing)
<br>
        
### 数据增广(Data Augmentation)

**TeachAugment: Data Augmentation Optimization Using Teacher Knowledge(使用教师知识进行数据增强优化)**<br>
[paper](https://arxiv.org/abs/2202.12513) | [code](https://github.com/DensoITLab/TeachAugment)<br>
<br>
**3D Common Corruptions and Data Augmentation(3D 常见损坏和数据增强)**<br>
*keywords: Data Augmentation, Image restoration, Photorealistic image synthesis*<br>
[paper](https://arxiv.org/abs/2203.01441)<br>
<br>

### 图像聚类(Image Clustering)

**RAMA: A Rapid Multicut Algorithm on GPU(GPU 上的快速多切算法)**<br>
[paper](https://arxiv.org/abs/2109.01838) | [code](https://github.com/pawelswoboda/RAMA)<br>
<br>

### 图像压缩(Image Compression)

**The Devil Is in the Details: Window-based Attention for Image Compression(细节中的魔鬼：图像压缩的基于窗口的注意力)**<br>
[paper](https://arxiv.org/abs/2203.08450) | [code](https://github.com/Googolxx/STF)<br>
<br>
**Neural Data-Dependent Transform for Learned Image Compression(用于学习图像压缩的神经数据相关变换)**<br>
[paper](https://arxiv.org/abs/2203.04963) | [code](https://dezhao-wang.github.io/Neural-Syntax-Website/)<br>
<br>

### 异常检测(Anomaly Detection)

**Generative Cooperative Learning for Unsupervised Video Anomaly Detection(用于无监督视频异常检测的生成式协作学习)**<br>
[paper](https://arxiv.org/abs/2203.03962)<br>
<br>
**Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection(用于异常检测的自监督预测卷积注意力块)(论文暂未上传)**<br>
[paper](https://arxiv.org/abs/2111.09099) | [code](https://github.com/ristea/sspcab)<br>
<br>

<a name="ModelTraining"></a>
## 模型训练/泛化(Model Training/Generalization)
<br>
        
### 模型训练/泛化(Model Training/Generalization)

**Can Neural Nets Learn the Same Model Twice? Investigating Reproducibility and Double Descent from the Decision Boundary Perspective(神经网络可以两次学习相同的模型吗？ 从决策边界的角度研究可重复性和双重下降)**<br>
[paper](https://arxiv.org/abs/2203.08124) | [code](https://github.com/somepago/dbViz)<br>
<br>
**Towards Efficient and Scalable Sharpness-Aware Minimization(迈向高效和可扩展的锐度感知最小化)**<br>
*keywords: Sharp Local Minima, Large-Batch Training*<br>
[paper](https://arxiv.org/abs/2203.02714)<br>
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

### 噪声标签(Noisy Label)

**Scalable Penalized Regression for Noise Detection in Learning with Noisy Labels(Scalable Penalized Regression for Noise Detection in Learning with Noisy Labels)**<br>
[paper](https://arxiv.org/abs/2203.07788) | [code](https://github.com/Yikai-Wang/SPR-LNL)<br>
<br>

### 长尾分布(Long-Tailed Distribution)

**Targeted Supervised Contrastive Learning for Long-Tailed Recognition(用于长尾识别的有针对性的监督对比学习)**<br>
*keywords: Long-Tailed Recognition(长尾识别), Contrastive Learning(对比学习)*<br>
[paper](https://arxiv.org/pdf/2111.13998.pdf)<br>
<br>

<a name="matching"></a>
## 图像特征提取与匹配(Image feature extraction and matching)
<br>
        
### 图像特征提取与匹配(Image feature extraction and matching)

**Probabilistic Warp Consistency for Weakly-Supervised Semantic Correspondences(弱监督语义对应的概率扭曲一致性)**<br>
[paper](https://arxiv.org/abs/2203.04279) | [code](https://github.com/PruneTruong/DenseMatching)<br>
<br>

<a name="VisualRL"></a>
## 视觉表征学习(Visual Representation Learning)
<br>
        
### 视觉表征学习(Visual Representation Learning)

**Exploring Set Similarity for Dense Self-supervised Representation Learning(探索密集自监督表示学习的集合相似性)**<br>
[paper](https://arxiv.org/abs/2107.08712)<br>
<br>
**Motion-aware Contrastive Video Representation Learning via Foreground-background Merging(通过前景-背景合并的运动感知对比视频表示学习)**<br>
[paper](https://arxiv.org/abs/2109.15130) | [code](https://github.com/Mark12Ding/FAME)<br>
<br>

<a name="MMLearning"></a>
## 多模态学习(Multi-Modal Learning)
<br>
        
### 多模态学习(Multi-Modal Learning)

**MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound(通过视觉、语言和声音的神经脚本知识)**<br>
[paper](https://arxiv.org/abs/2201.02639)<br>
<br>

### 视觉-语言（Vision-language）

**Pseudo-Q: Generating Pseudo Language Queries for Visual Grounding(为视觉基础生成伪语言查询)**<br>
[paper](https://arxiv.org/abs/2203.08481) | [code](https://github.com/LeapLabTHU/Pseudo-Q)<br>
<br>
**Conditional Prompt Learning for Vision-Language Models(视觉语言模型的条件提示学习)**<br>
[paper](https://arxiv.org/abs/2203.05557) | [code](https://github.com/KaiyangZhou/CoOp)<br>
<br>
**NLX-GPT: A Model for Natural Language Explanations in Vision and Vision-Language Tasks(视觉和视觉语言任务中的自然语言解释模型)**<br>
[paper](https://arxiv.org/abs/2203.05081) | [code](https://github.com/fawazsammani/nlxgpt)<br>
<br>
**L-Verse: Bidirectional Generation Between Image and Text(图像和文本之间的双向生成) **(Oral Presentation)****<br>
[paper](https://arxiv.org/abs/2111.11133)<br>
<br>
**HairCLIP: Design Your Hair by Text and Reference Image(通过文本和参考图像设计你的头发)**<br>
*keywords: Language-Image Pre-Training (CLIP), Generative Adversarial Networks*<br>
[paper](https://arxiv.org/abs/2112.05142)<br>
<br>
**CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields(文本和图像驱动的神经辐射场操作)**<br>
*keywords: NeRF,  Image Generation and Manipulation, Language-Image Pre-Training (CLIP)*<br>
[paper](https://arxiv.org/abs/2112.05139) | [code](https://cassiepython.github.io/clipnerf/)<br>
<br>
**Vision-Language Pre-Training with Triple Contrastive Learning(三重对比学习的视觉语言预训练)**<br>
*keywords: Vision-language representation learning, Contrastive Learning*<br>
[paper](https://arxiv.org/abs/2202.10401) | [code](https://github.com/uta-smile/TCL;)<br>
<br>

<a name="Vision-basedPrediction"></a>
## 视觉预测(Vision-based Prediction)
<br>
        
### 视觉预测(Vision-based Prediction)

**On Adversarial Robustness of Trajectory Prediction for Autonomous Vehicles(自动驾驶汽车轨迹预测的对抗鲁棒性)**<br>
[paper](https://arxiv.org/abs/2201.05057) | [code](https://github.com/zqzqz/AdvTrajectoryPrediction)<br>
<br>
**Adaptive Trajectory Prediction via Transferable GNN(基于可迁移 GNN 的自适应轨迹预测)**<br>
[paper](https://arxiv.org/abs/2203.05046)<br>
<br>
**Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective(迈向稳健和自适应运动预测：因果表示视角)**<br>
[paper](https://arxiv.org/abs/2111.14820) | [code](https://github.com/vita-epfl/causalmotion)<br>
<br>
**How many Observations are Enough? Knowledge Distillation for Trajectory Forecasting(多少个观察就足够了？ 轨迹预测的知识蒸馏)**<br>
*keywords: Knowledge Distillation, trajectory forecasting*<br>
[paper](https://arxiv.org/abs/2203.04781)<br>
<br>
**Motron: Multimodal Probabilistic Human Motion Forecasting(多模式概率人体运动预测)**<br>
[paper](https://arxiv.org/abs/2203.04132)<br>
<br>

<a name="Dataset"></a>
## 数据集(Dataset)
<br>
        
### 数据集(Dataset)

**FERV39k: A Large-Scale Multi-Scene Dataset for Facial Expression Recognition in Videos(用于视频中面部表情识别的大规模多场景数据集)**<br>
[paper](https://arxiv.org/abs/2203.09463)<br>
<br>
**Ego4D: Around the World in 3,000 Hours of Egocentric Video(3000 小时以自我为中心的视频环游世界)**<br>
[paper](https://arxiv.org/abs/2110.07058)<br>
<br>
**GrainSpace: A Large-scale Dataset for Fine-grained and Domain-adaptive Recognition of Cereal Grains(用于细粒度和域自适应识别谷物的大规模数据集)**<br>
[paper](https://arxiv.org/abs/2203.05306)<br>
<br>
**Kubric: A scalable dataset generator(Kubric：可扩展的数据集生成器)**<br>
[paper](https://arxiv.org/abs/2203.03570) | [code](https://github.com/google-research/kubric)<br>
<br>
**A Large-scale Comprehensive Dataset and Copy-overlap Aware Evaluation Protocol for Segment-level Video Copy Detection(用于分段级视频复制检测的大规模综合数据集和复制重叠感知评估协议)**<br>
[paper](https://arxiv.org/abs/2203.02654)<br>
<br>

<a name="ActiveLearning"></a>
## 主动学习(Active Learning)
<br>
        
### 主动学习(Active Learning)

**Active Learning by Feature Mixing(通过特征混合进行主动学习)**<br>
[paper](https://arxiv.org/abs/2203.07034) | [code](https://github.com/Haoqing-Wang/InfoCL)<br>
<br>

<a name="None"></a>
## 小样本学习/零样本学习(Few-shot Learning/Zero-shot Learning)
<br>
        
### 小样本学习/零样本学习(Few-shot Learning/Zero-shot Learning)

**Learning to Affiliate: Mutual Centralized Learning for Few-shot Classification(小样本分类的相互集中学习)**<br>
[paper](https://arxiv.org/abs/2106.05517)<br>
<br>
**MSDN: Mutually Semantic Distillation Network for Zero-Shot Learning(用于零样本学习的相互语义蒸馏网络)**<br>
*keywords: Zero-Shot Learning,  Knowledge Distillation*<br>
[paper](https://arxiv.org/abs/2203.03137) | [code](https://github.com/shiming-chen/MSDN)<br>
<br>

<a name="ContinualLearning"></a>
## 持续学习(Continual Learning/Life-long Learning)
<br>
        
### 持续学习(Continual Learning/Life-long Learning)

**On Generalizing Beyond Domains in Cross-Domain Continual Learning(关于跨域持续学习中的域外泛化)**<br>
[paper](https://arxiv.org/abs/2203.03970)<br>
<br>

<a name="None"></a>
## 场景图(Scene Graph)
<br>
        
### 场景图生成(Scene Graph Generation)

**Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs(将视频场景图重新格式化为时间二分图)**<br>
*keywords: Video Scene Graph Generation, Transformer, Video Grounding*<br>
[paper](https://arxiv.org/abs/2112.04222) | [code](https://github.com/Dawn-LX/VidVRD-tracklets)<br>
<br>

<a name="VisualLocalization"></a>
## 视觉定位/位姿估计(Visual Localization/Pose Estimation)
<br>
        
### 视觉定位/位姿估计(Visual Localization/Pose Estimation)

**ZebraPose: Coarse to Fine Surface Encoding for 6DoF Object Pose Estimation(用于 6DoF 对象姿态估计的粗到细表面编码)**<br>
[paper](https://arxiv.org/abs/2203.09418)<br>
<br>
**Object Localization under Single Coarse Point Supervision(单粗点监督下的目标定位)**<br>
[paper](https://arxiv.org/abs/2203.09338) | [code](https://github.com/ucas-vg/PointTinyBenchmark/)<br>
<br>
**CrossLoc: Scalable Aerial Localization Assisted by Multimodal Synthetic Data(多模式合成数据辅助的可扩展空中定位)**<br>
[paper](https://arxiv.org/abs/2112.09081) | [code](https://github.com/TOPO-EPFL/CrossLoc)<br>
<br>
**GPV-Pose: Category-level Object Pose Estimation via Geometry-guided Point-wise Voting(通过几何引导的逐点投票进行类别级对象位姿估计)**<br>
[paper](https://arxiv.org/abs/2203.07918) | [code](https://github.com/lolrudy/GPV_Pose)<br>
<br>
**CPPF: Towards Robust Category-Level 9D Pose Estimation in the Wild(CPPF：在野外实现稳健的类别级 9D 位姿估计)**<br>
[paper](https://arxiv.org/abs/2203.03089) | [code](https://github.com/qq456cvb/CPPF)<br>
<br>
**OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation(用于基于深度的 6D 对象位姿估计的对象视点编码)**<br>
[paper](https://arxiv.org/abs/2203.01072) | [code](https://github.com/dingdingcai/OVE6D-pose)<br>
<br>
**Spatial Commonsense Graph for Object Localisation in Partial Scenes(局部场景中对象定位的空间常识图)**<br>
[paper](https://arxiv.org/abs/2203.05380) | [code](https://github.com/FGiuliari/SpatialCommonsenseGraph-Dataset)<br>
<br>

<a name="VisualReasoning"></a>
## 视觉推理/视觉问答(Visual Reasoning/VQA)
<br>
        
### 视觉推理/视觉问答(Visual Reasoning/VQA)

**MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering(基于知识的视觉问答的多模态知识提取与积累)**<br>
[paper](https://arxiv.org/abs/2203.09138) | [code](https://github.com/AndersonStra/MuKEA)<br>
<br>
**REX: Reasoning-aware and Grounded Explanation(推理意识和扎根的解释)**<br>
[paper](https://arxiv.org/abs/2203.06107) | [code](https://github.com/szzexpoi/rex)<br>
<br>

<a name="ImageClassification"></a>
## 图像分类(Image Classification)
<br>
        
### 图像分类(Image Classification)

**GlideNet: Global, Local and Intrinsic based Dense Embedding NETwork for Multi-category Attributes Prediction(用于多类别属性预测的基于全局、局部和内在的密集嵌入网络)**<br>
*keywords: multi-label classification*<br>
[paper](https://arxiv.org/abs/2203.03079) | [code](https://github.com/kareem-metwaly/glidenet)<br>
<br>

<a name="domain"></a>
## 迁移学习/domain/自适应(Transfer Learning/Domain Adaptation)
<br>
        
### 迁移学习/domain/自适应(Transfer Learning/Domain Adaptation)

**Category Contrast for Unsupervised Domain Adaptation in Visual Tasks(视觉任务中无监督域适应的类别对比)**<br>
[paper](https://arxiv.org/abs/2106.02885)<br>
<br>
**Learning Distinctive Margin toward Active Domain Adaptation(向主动领域适应学习独特的边际)**<br>
[paper](https://arxiv.org/abs/2203.05738) | [code](https://github.com/TencentYoutuResearch/ActiveLearning-SDM)<br>
<br>
**How Well Do Sparse Imagenet Models Transfer?(稀疏 Imagenet 模型的迁移效果如何？)**<br>
[paper](https://arxiv.org/abs/2111.13445)<br>
<br>
**A Simple Multi-Modality Transfer Learning Baseline for Sign Language Translation(用于手语翻译的简单多模态迁移学习基线)**<br>
[paper](https://arxiv.org/abs/2203.04287)<br>
<br>
**Weakly Supervised Object Localization as Domain Adaption(作为域适应的弱监督对象定位)**<br>
*keywords: Weakly Supervised Object Localization(WSOL), Multi-instance learning based WSOL, Separated-structure based WSOL, Domain Adaption*<br>
[paper](https://arxiv.org/abs/2203.01714) | [code](https://github.com/zh460045050/DA-WSOL_CVPR2022)<br>
<br>

<a name="MetricLearning"></a>
## 度量学习(Metric Learning)
<br>
        
### 度量学习(Metric Learning)

**Non-isotropy Regularization for Proxy-based Deep Metric Learning(基于代理的深度度量学习的非各向同性正则化)**<br>
[paper](https://arxiv.org/abs/2203.08547) | [code](https://github.com/ExplainableML/NonIsotropicProxyDML)<br>
<br>
**Integrating Language Guidance into Vision-based Deep Metric Learning(将语言指导集成到基于视觉的深度度量学习中)**<br>
[paper](https://arxiv.org/abs/2203.08543) | [code](https://github.com/ExplainableML/LanguageGuidance_for_DML)<br>
<br>
**Enhancing Adversarial Robustness for Deep Metric Learning(增强深度度量学习的对抗鲁棒性)**<br>
*keywords: Adversarial Attack, Adversarial Defense, Deep Metric Learning*<br>
[paper](https://arxiv.org/pdf/2203.01439.pdf)<br>
<br>

<a name="ContrastiveLearning"></a>
## 对比学习(Contrastive Learning)
<br>
        
### 对比学习(Contrastive Learning)

**Rethinking Minimal Sufficient Representation in Contrastive Learning(重新思考对比学习中的最小充分表示)**<br>
[paper](https://arxiv.org/abs/2203.07004) | [code](https://github.com/Haoqing-Wang/InfoCL)<br>
<br>
**Selective-Supervised Contrastive Learning with Noisy Labels(带有噪声标签的选择性监督对比学习)**<br>
[paper](https://arxiv.org/abs/2203.04181) | [code](https://github.com/ShikunLi/Sel-CL)<br>
<br>
**HCSC: Hierarchical Contrastive Selective Coding(分层对比选择性编码)**<br>
*keywords: Self-supervised Representation Learning, Deep Clustering, Contrastive Learning*<br>
[paper](https://arxiv.org/abs/2202.00455) | [code](https://github.com/gyfastas/HCSC)<br>
<br>
**Crafting Better Contrastive Views for Siamese Representation Learning(为连体表示学习制作更好的对比视图)**<br>
[paper](https://arxiv.org/pdf/2202.03278.pdf) | [code](https://github.com/xyupeng/ContrastiveCrop)<br>
<br>

<a name="IncrementalLearning"></a>
## 增量学习(Incremental Learning)
<br>
        
### 增量学习(Incremental Learning)

**Forward Compatible Few-Shot Class-Incremental Learning(前后兼容的小样本类增量学习)**<br>
[paper](https://arxiv.org/abs/2203.06953) | [code](https://github.com/zhoudw-zdw/CVPR22-Fact)<br>
<br>
**Self-Sustaining Representation Expansion for Non-Exemplar Class-Incremental Learning(非示例类增量学习的自我维持表示扩展)**<br>
[paper](https://arxiv.org/abs/2203.06359)<br>
<br>

<a name="MetaLearning"></a>
## 元学习(Meta Learning)
<br>
        
### 元学习(Meta Learning)

**What Matters For Meta-Learning Vision Regression Tasks?(元学习视觉回归任务的重要性是什么？)**<br>
[paper](https://arxiv.org/abs/2203.04905)<br>
<br>

<a name="Robotic"></a>
## 机器人(Robotic)
<br>
        
### 机器人(Robotic)

**Coarse-to-Fine Q-attention: Efficient Learning for Visual Robotic Manipulation via Discretisation(通过离散化实现视觉机器人操作的高效学习)**<br>
[paper](https://arxiv.org/abs/2106.12534) | [code](https://github.com/stepjam/ARM)<br>
<br>
**IFOR: Iterative Flow Minimization for Robotic Object Rearrangement(IFOR：机器人对象重排的迭代流最小化)**<br>
[paper](https://arxiv.org/pdf/2202.00732.pdf)<br>
<br>

<a name="self-supervisedlearning"></a>
## 自监督学习/半监督学习/无监督学习(Self-supervised Learning/Semi-supervised Learning)
<br>
        
### 自监督学习/半监督学习/无监督学习(Self-supervised Learning/Semi-supervised Learning)

**SimMatch: Semi-supervised Learning with Similarity Matching(具有相似性匹配的半监督学习)**<br>
[paper](https://arxiv.org/abs/2203.06915) | [code](https://github.com/KyleZheng1997/simmatch)<br>
<br>
**Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements(一个完全无监督的框架，用于从噪声和部分测量中学习图像)**<br>
[paper](https://arxiv.org/abs/2111.12855) | [code](https://github.com/edongdongchen/REI)<br>
<br>
**UniVIP: A Unified Framework for Self-Supervised Visual Pre-training(自监督视觉预训练的统一框架)**<br>
[paper](https://arxiv.org/abs/2203.06965)<br>
<br>
**Class-Aware Contrastive Semi-Supervised Learning(类感知对比半监督学习)**<br>
*keywords: Semi-Supervised Learning, Self-Supervised Learning, Real-World Unlabeled Data Learning*<br>
[paper](https://arxiv.org/abs/2203.02261)<br>
<br>
**A study on the distribution of social biases in self-supervised learning visual models(自监督学习视觉模型中social biases分布的研究)**<br>
[paper](https://arxiv.org/pdf/2203.01854.pdf)<br>
<br>

<a name="interpretability"></a>
## 神经网络可解释性(Neural Network Interpretability)
<br>
        
### 神经网络可解释性(Neural Network Interpretability)

**Do Explanations Explain? Model Knows Best(解释解释吗？ 模型最清楚)**<br>
[paper](https://arxiv.org/abs/2203.02269)<br>
<br>
**Interpretable part-whole hierarchies and conceptual-semantic relationships in neural networks(神经网络中可解释的部分-整体层次结构和概念语义关系)**<br>
[paper](https://arxiv.org/abs/2203.03282)<br>
<br>

<a name="CrowdCounting"></a>
## 图像计数(Image Counting)
<br>
        
### 图像计数(Image Counting)

**Represent, Compare, and Learn: A Similarity-Aware Framework for Class-Agnostic Counting(表示、比较和学习：用于类不可知计数的相似性感知框架)**<br>
[paper](https://arxiv.org/abs/2203.08354) | [code](https://github.com/flyinglynx/Bilinear-Matching-Network)<br>
<br>
**Boosting Crowd Counting via Multifaceted Attention(通过多方面注意提高人群计数)**<br>
[paper](https://arxiv.org/pdf/2203.02636.pdf) | [code](https://github.com/LoraLinH/Boosting-Crowd-Counting-via-Multifaceted-Attention)<br>
<br>

<a name="None"></a>
## 联邦学习(Federated Learning)
<br>
        
### 联邦学习(Federated Learning)

**Fine-tuning Global Model via Data-Free Knowledge Distillation for Non-IID Federated Learning(通过非 IID 联邦学习的无数据知识蒸馏微调全局模型)**<br>
[paper](https://arxiv.org/abs/2203.09249)<br>
<br>
**Differentially Private Federated Learning with Local Regularization and Sparsification(局部正则化和稀疏化的差分私有联邦学习)**<br>
[paper](https://arxiv.org/abs/2203.03106)<br>
<br>

<a name="None"></a>
## 其他
<br>
        
### 其他

**L-Verse: Bidirectional Generation Between Image and Text(图像和文本之间的双向生成) **(视觉语言表征学习)****<br>
[paper](https://arxiv.org/abs/2111.11133) | [code](https://github.com/nie-lang/DeepRectangling)<br>
<br>

<a name="None"></a>
##  Backbone
<br>
        
###  Backbone

**MPViT : Multi-Path Vision Transformer for Dense Prediction**<br>
[paper](https://arxiv.org/abs/2112.11010) | [code](https://github.com/youngwanLEE/MPViT)<br>
<br>

<a name="None"></a>
##  CLIP
<br>
        
###  CLIP

**PointCLIP: Point Cloud Understanding by CLIP**<br>
[paper](https://arxiv.org/abs/2112.02413) | [code](https://github.com/ZrrSkywalker/PointCLIP)<br>
<br>
**Blended Diffusion for Text-driven Editing of Natural Images**<br>
[paper](https://arxiv.org/abs/2111.14818) | [code](https://github.com/omriav/blended-diffusion)<br>
<br>

<a name="None"></a>
##  GAN
<br>
        
###  GAN

**SemanticStyleGAN: Learning Compositional Generative Priors for Controllable Image Synthesis and Editing**<br>
[paper](https://arxiv.org/abs/2112.02236)<br>
<br>

<a name="None"></a>
##  NAS
<br>
        
###  NAS

**ISNAS-DIP: Image-Specific Neural Architecture Search for Deep Image Prior**<br>
[paper](https://arxiv.org/abs/2111.15362) | [code](None)<br>
<br>

<a name="None"></a>
##  NeRF
<br>
        
###  NeRF

**Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields**<br>
[paper](https://arxiv.org/abs/2111.12077)<br>
<br>
**NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images**<br>
[paper](https://arxiv.org/abs/2111.13679)<br>
<br>
**Urban Radiance Fields**<br>
[paper](https://arxiv.org/abs/2111.14643)<br>
<br>
**Pix2NeRF: Unsupervised Conditional π-GAN for Single Image to Neural Radiance Fields Translation**<br>
[paper](https://arxiv.org/abs/2202.13162) | [code](https://github.com/HexagonPrime/Pix2NeRF)<br>
<br>
**HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video**<br>
[paper](https://arxiv.org/abs/2201.04127)<br>
<br>

<a name="None"></a>
##  Visual Transformer
<br>
        
###  Backbone

**MPViT : Multi-Path Vision Transformer for Dense Prediction**<br>
[paper](https://arxiv.org/abs/2112.11010) | [code](https://github.com/youngwanLEE/MPViT)<br>
<br>

###  应用(Application)

**Language-based Video Editing via Multi-Modal Multi-Level Transformer**<br>
[paper](https://arxiv.org/abs/2104.01122) | [code](None)<br>
<br>
**Embracing Single Stride 3D Object Detector with Sparse Transformer**<br>
[paper](https://arxiv.org/abs/2112.06375) | [code](https://github.com/TuSimple/SST)<br>
<br>
**Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction**<br>
[paper](https://arxiv.org/abs/2111.07910) | [code](https://github.com/caiyuanhao1998/MST)<br>
<br>
**Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling**<br>
[paper](https://arxiv.org/abs/2111.14819) | [code](https://github.com/lulutang0608/Point-BERT)<br>
<br>
**GroupViT: Semantic Segmentation Emerges from Text Supervision**<br>
[paper](https://arxiv.org/abs/2202.11094)<br>
<br>
**Splicing ViT Features for Semantic Appearance Transfer**<br>
[paper](https://arxiv.org/abs/2201.00424) | [code](https://github.com/omerbt/Splice)<br>
<br>
**Self-supervised Video Transformer**<br>
[paper](https://arxiv.org/abs/2112.01514) | [code](https://github.com/kahnchana/svt)<br>
<br>

<a name="None"></a>
##  数据增强(Data Augmentation)
<br>
        
###  数据增强(Data Augmentation)

**AlignMix: Improving representation by interpolating aligned features**<br>
[paper](https://arxiv.org/abs/2103.15375) | [code](None)<br>
<br>

<a name="None"></a>
##  语义分割(Semantic Segmentation)
<br>
        
###  无监督语义分割

**GroupViT: Semantic Segmentation Emerges from Text Supervision**<br>
[paper](https://arxiv.org/abs/2202.11094)<br>
<br>

<a name="None"></a>
##  实例分割(Instance Segmentation)
<br>
        
###  自监督实例分割

**FreeSOLO: Learning to Segment Objects without Annotations**<br>
[paper](https://arxiv.org/abs/2202.12181) | [code](None)<br>
<br>

<a name="None"></a>
##  视频理解(Video Understanding)
<br>
        
###  视频理解(Video Understanding)

**Self-supervised Video Transformer**<br>
[paper](https://arxiv.org/abs/2112.01514) | [code](https://github.com/kahnchana/svt)<br>
<br>

<a name="None"></a>
##  图像编辑(Image Editing)
<br>
        
###  图像编辑(Image Editing)

**Blended Diffusion for Text-driven Editing of Natural Images**<br>
[paper](https://arxiv.org/abs/2111.14818) | [code](https://github.com/omriav/blended-diffusion)<br>
<br>
**SemanticStyleGAN: Learning Compositional Generative Priors for Controllable Image Synthesis and Editing**<br>
[paper](https://arxiv.org/abs/2112.02236)<br>
<br>

<a name="None"></a>
##  Low-level Vision
<br>
        
###  Low-level Vision

**ISNAS-DIP: Image-Specific Neural Architecture Search for Deep Image Prior**<br>
[paper](https://arxiv.org/abs/2111.15362) | [code](None)<br>
<br>

<a name="None"></a>
##  超分辨率(Super-Resolution)
<br>
        
###  图像超分辨率(Image Super-Resolution)

**Learning the Degradation Distribution for Blind Image Super-Resolution**<br>
[paper](https://arxiv.org/abs/2203.04962) | [code](https://github.com/greatlog/UnpairedSR)<br>
<br>

###  视频超分辨率(Video Super-Resolution)

**BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment**<br>
[paper](https://arxiv.org/abs/2104.13371) | [code](https://github.com/ckkelvinchan/BasicVSR_PlusPlus)<br>
<br>

<a name="None"></a>
##  3D点云(3D Point Cloud)
<br>
        
###  3D点云(3D Point Cloud)

**Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling**<br>
[paper](https://arxiv.org/abs/2111.14819) | [code](https://github.com/lulutang0608/Point-BERT)<br>
<br>
**PointCLIP: Point Cloud Understanding by CLIP**<br>
[paper](https://arxiv.org/abs/2112.02413) | [code](https://github.com/ZrrSkywalker/PointCLIP)<br>
<br>

<a name="None"></a>
##  3D目标检测(3D Object Detection)
<br>
        
###  3D目标检测(3D Object Detection)

**Embracing Single Stride 3D Object Detector with Sparse Transformer**<br>
[paper](https://arxiv.org/abs/2112.06375) | [code](https://github.com/TuSimple/SST)<br>
<br>

<a name="None"></a>
##  3D语义场景补全(3D Semantic Scene Completion)
<br>
        
###  3D语义场景补全(3D Semantic Scene Completion)

**MonoScene: Monocular 3D Semantic Scene Completion**<br>
[paper](https://arxiv.org/abs/2112.00726) | [code](https://github.com/cv-rits/MonoScene)<br>
<br>

<a name="None"></a>
##  3D重建(3D Reconstruction)
<br>
        
###  3D重建(3D Reconstruction)

**BANMo: Building Animatable 3D Neural Models from Many Casual Videos**<br>
[paper](https://arxiv.org/abs/2112.12761) | [code](https://github.com/facebookresearch/banmo)<br>
<br>

<a name="None"></a>
##  深度估计(Depth Estimation)
<br>
        
###  单目深度估计

**Toward Practical Self-Supervised Monocular Indoor Depth Estimation**<br>
[paper](https://arxiv.org/abs/2112.02306) | [code](None)<br>
<br>

<a name="None"></a>
##  人群计数(Crowd Counting)
<br>
        
###  人群计数(Crowd Counting)

**Leveraging Self-Supervision for Cross-Domain Crowd Counting**<br>
[paper](https://arxiv.org/abs/2103.16291) | [code](None)<br>
<br>

<a name="None"></a>
##  医学图像(Medical Image)
<br>
        
###  医学图像(Medical Image)

**BoostMIS: Boosting Medical Image Semi-supervised Learning with Adaptive Pseudo Labeling and Informative Active Annotation**<br>
[paper](https://arxiv.org/abs/2203.02533) | [code](None)<br>
<br>

<a name="None"></a>
##  场景图生成(Scene Graph Generation)
<br>
        
###  场景图生成(Scene Graph Generation)

**SGTR: End-to-end Scene Graph Generation with Transformer**<br>
[paper](https://arxiv.org/abs/2112.12970) | [code](None)<br>
<br>

<a name="None"></a>
##  高光谱图像重建(Hyperspectral Image Reconstruction)
<br>
        
###  高光谱图像重建(Hyperspectral Image Reconstruction)

**Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction**<br>
[paper](https://arxiv.org/abs/2111.07910) | [code](https://github.com/caiyuanhao1998/MST)<br>
<br>

<a name="None"></a>
##  水印(Watermarking)
<br>
        
###  水印(Watermarking)

**Deep 3D-to-2D Watermarking: Embedding Messages in 3D Meshes and Extracting Them from 2D Renderings**<br>
[paper](https://arxiv.org/abs/2104.13450) | [code](None)<br>
<br>

<a name="None"></a>
##  数据集(Datasets)
<br>
        
###  数据集(Datasets)

**It's About Time: Analog Clock Reading in the Wild**<br>
[paper](https://arxiv.org/abs/2111.09162) | [code](https://github.com/charigyang/itsabouttime)<br>
<br>
**Toward Practical Self-Supervised Monocular Indoor Depth Estimation**<br>
[paper](https://arxiv.org/abs/2112.02306) | [code](None)<br>
<br>

<a name="None"></a>
##  新任务(New Task)
<br>
        
###  新任务(New Task)

**Language-based Video Editing via Multi-Modal Multi-Level Transformer**<br>
[paper](https://arxiv.org/abs/2104.01122) | [code](None)<br>
<br>
**It's About Time: Analog Clock Reading in the Wild**<br>
[paper](https://arxiv.org/abs/2111.09162) | [code](https://github.com/charigyang/itsabouttime)<br>
<br>
**Splicing ViT Features for Semantic Appearance Transfer**<br>
[paper](https://arxiv.org/abs/2201.00424) | [code](https://github.com/omerbt/Splice)<br>
<br>
