import os

latex_content = r"""\documentclass[UTF8,a4paper,12pt]{ctexart}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{float}
\usepackage{listings}
\usepackage{xcolor}

\geometry{left=2.2cm,right=2.2cm,top=2.3cm,bottom=2.3cm}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.35em}
\renewcommand{\arraystretch}{1.15}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.96,0.96,0.96}

\lstset{
    language=Python,
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=4
}

\title{\textbf{机器学习实验报告}\\[1ex]\Large{实验八：生成模型对比分析}}
\author{实验人员: 柯力洲}
\date{\today}

\begin{document}
\maketitle

\vspace*{-1cm}
\begin{center}
    数学实验中心
\end{center}
\vspace*{1em}

\section*{一、实验内容 \& 二、实验描述}
对比测试三种主流图像生成范式（Diffusion、Flow Matching 和 Autoregressive 模型），在同一套训练环境和硬件配置下，对它们的生成质量与收敛开销进行对比评价。并引入预训练的先进基准模型（SOTA，如 DDPM CIFAR-10 官方开源版本）作为外部参照物。
所有的模型均在 CIFAR-10 数据集上训练和采样，输入分辨率设定为 $32\times32$。硬件采用 8 张 24G 显存 GPU 并行训练。使用的核心评估指标为 FID（Fréchet Inception Distance）。

\section*{三、三种生成范式的原理解析}

\subsection*{1. 扩散模型 (Diffusion Model)}
扩散模型由前向加噪（Forward Process）和逆向去噪（Reverse Process）组成。前向过程通过马尔可夫链向数据中逐步添加高斯噪声，直到图片完全变为各向同性的标准高斯分布。其第 $t$ 步的公式可表示为：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$。逆向过程则是利用一个 U-Net 神经网络来预测每一步加入的噪声，通过朗之万退火或 DDPM 的变分推断途径，从纯噪声还原为清晰图像。

\subsection*{2. 流匹配模型 (Flow Matching)}
Flow Matching 是一种利用常微分方程 (ODE) 构建连续标准化流 (Normalizing Flow) 的无模拟目标的生成框架。它将数据分布和高斯分布设为两个端点，利用最优传输理论直接构建从分布 $p_0$ 指向 $p_1$ 的固定目标向量场（Target Vector Field）。网络在训练期间只需拟合该向量场即可：$\mathcal{L} = \mathbb{E}_{t, x_1, x_0} \| v_\theta(x_t, t) - (x_1 - x_0) \|^2$。基于此框架生成的路径更为平滑，训练收敛比传统的 SDE 方法更高效。

\subsection*{3. 自回归模型 (Autoregressive Generator)}
自回归模型深受大语言模型启发。图像不再被视为连续的二维矩阵，而是展平成一维的离散序列。目标函数变成联合概率分解条件概率的连乘：$P(x_1, \dots, x_N) = \prod_{i=1}^{N} P(x_i | x_{<i})$。模型基于掩码注意力机制（Causal Mask），严格按照从左到右、从上到下的顺序每次对单个像素的类别分布（或者潜在 Token 索引）进行多分类预测并采样，具有极强的联合分布拟合能力。

\section*{四、核心算法与详尽代码实现}

\subsection*{1. Diffusion Model (扩散模型)}
前向过程通过累加的 $\alpha$ 折算直接得到任意时间的含噪样本，模型负责捕捉加入的噪声模式。

\begin{lstlisting}[language=Python]
# 前向加噪以及损失计算 (Forward Process & Training Objective)
def diffusion_loss(denoise_model, x_start, t):
    noise = torch.randn_like(x_start) # 1. 采样标准高斯噪声
    # 2. 根据重参数化公式，提取当前步 t 对应的系数
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    # 3. 计算 t 时刻的含噪图像 x_t
    x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # 4. Neural Network (如 U-Net) 预测此时的噪声模式
    predicted_noise = denoise_model(x_noisy, t)
    
    # 5. MSE 优化
    return F.mse_loss(noise, predicted_noise)

# 逆向推断 (Reverse Sampling Loop)
@torch.no_grad()
def p_sample_loop(model, shape):
    img = torch.randn(shape, device=device) # 从纯噪声画布开始
    for i in reversed(range(0, timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        pred_noise = model(img, t)
        
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        beta = betas[t]
        
        # 6. DDPM 后验采样推断 x_{t-1}
        img = (1/torch.sqrt(alpha))*(img - ((1-alpha)/torch.sqrt(1-alpha_bar))*pred_noise)
        if i > 0:
            img = img + torch.sqrt(beta) * torch.randn_like(img)
    return img
\end{lstlisting}

\subsection*{2. Flow Matching (流匹配)}
极简的损失设计与完全基于 ODE 偏微分数值求解器的推断。

\begin{lstlisting}[language=Python]
# 线性插值与流场损失拟合 (Target Vector Matching Loss)
def flow_matching_loss(model, x_1):
    x_0 = torch.randn_like(x_1) # 随机噪声状态
    t = torch.rand((x_1.shape[0],)).to(x_1.device)
    t_expand = t.view(-1, 1, 1, 1)

    # 1. 构造沿时间 t 方向的直接线性演化路径
    x_t = (1 - t_expand) * x_0 + t_expand * x_1
    
    # 2. 计算先验理想标场 (Target Vector)
    target_v = x_1 - x_0 

    # 3. 拟合该状态的向量方向
    predicted_v = model(x_t, t)
    return F.mse_loss(predicted_v, target_v)

# 基于欧拉法的一阶求解推断 (Euler ODE Solver)
@torch.no_grad()
def sample_flow_matching(model, shape, steps=100):
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.ones(shape[0], device=device) * (i / steps)
        v_pred = model(x, t) # 利用网络预测流场切线方向
        
        # 4. 以欧拉方法累加场的变化
        x = x + v_pred * dt 
    return x
\end{lstlisting}

\subsection*{3. Autoregressive (自回归模型)}
将长序列因果预测直接实装于生成过程，需要大量循环步骤：

\begin{lstlisting}[language=Python]
# 通用分类交叉熵损失 (Categorical Cross Entropy)
def autoregressive_loss(model, x):
    # 输入展平序列 [Batch, Sequence_length]
    # 利用因果 Attention 机制预测下一个 Token 的分布
    logits = model(x[:, :-1]) 
    target = x[:, 1:]
    return F.cross_entropy(logits.reshape(-1, num_classes), target.reshape(-1))

# 自回归逐元素推断 (Sequential Raster-scan Generation)
@torch.no_grad()
def sample_autoregressive(model, shape):
    B, C, H, W = shape
    samples = torch.zeros(shape, device=device)
    # 1. 强制双循环模拟从上到下、左到右光栅扫描
    for r in range(H):
        for c in range(W):
            logits = model(samples)
            # 2. 每一步仅推断当前的这1个像素/通道分布
            probs = F.softmax(logits[:, :, r, c], dim=-1)
            # 3. Softmax 重采样固定概率获取具象值
            predicted_val = torch.multinomial(probs, num_samples=1).squeeze(-1)
            samples[:, :, r, c] = predicted_val
    return samples
\end{lstlisting}

\section*{五、生成过程动态可视化}
由于不同的范式其内在设计逻辑不同，反映到推断过程中，其生成路径表现出了截然不同的现象。
以下抽样直观展示了连续生成（Diffusion）与离散串行生成（Autoregressive）的过程差异。

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{figures/diffusion_process.png}
\caption{扩散模型的生成演化：全局同时从强高斯特化为具象像素，呈现无中生有、由模糊转为细节清晰（Coarse-to-fine）的直观效果。}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/ar_process.png}
\caption{自回归模型的生成演化：因依赖此前已生成的上下文信息，呈现按照几何顺序（如光栅排版）被局部填充满画面的块状补全过程。}
\end{figure}

\section*{六、实验结果图表展示与对比}

\begin{table}[H]
\centering
\caption{各生成模型在同等数据流下的量化表现对比（FID 越低意味着逼近真实分布）}
\begin{tabular}{lccc}
\toprule
模型层类 & 测试集 FID $\downarrow$ & 训练收敛时间(分钟) & 网络参数量(M) \\
\midrule
自训 Diffusion & 39.97 & 59.56 & 65.85 \\
自训 Flow Matching & 44.34 & 58.25 & 65.85 \\
自训 Autoregressive & 229.73 & 15.26 & 16.64 \\
SOTA (DDPM Cifar10) & 3.17 (理想天花板) & 不适用(预训练) & 35.70 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/fid_barplot.png}
    \caption{自训模型 FID 柱状比较}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/sota_samples.png}
    \caption{SOTA参考网格推演输出}
\end{subfigure}
\caption{量化指标对比与参照标杆图比对体系。}
\end{figure}

\section*{七、遇到的问题及其解决措施}
\begin{enumerate}
    \item \textbf{多卡并行的 OOM 显存碎片困境}：
    在初期运行 multi\_gpu\_8x24g 发送 batch 时偶发碎片分配阻断。引入环境中环境变量配置 \texttt{PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True} ，平滑释放了未利用分块化解瓶颈。
    \item \textbf{引入外部参照时遇网络阻断}：
    由于原生平台联通延迟多次造成 `huggingface\_hub` 握手超时。本地手动切换为源节点：\texttt{export HF\_ENDPOINT=https://hf-mirror.com} 并建立独立的 \texttt{benchmark\_sota.py} 进行测频采样。最后一次出图耗时约稳定于 26 秒水平。
\end{enumerate}

\section*{八、实验评价与总结}
本实验在同一测试域覆盖了目前主流界三种图像生成方式并落实了代码全环节。其中连续型代表 Diffusion 及 Flow 展现出了对长序列复杂语义边缘的优秀控制水平，它们平滑收敛，尽管单步推理时间过长是不可规避弊端。离散化代表 AR 模型，则体现了训练速度快、小算力下的高效吞吐优势，但对于像素强依赖性也使得它未能捕捉大片域色块平衡。最后对齐开源社区的天花板后得到启发：模型架构虽具优劣之分，但海量算力与精密调参依然是促成应用级成果决定性壁垒。

\end{document}
"""

with open("/home/klz/report/exp8_gen_compare/report8.tex", "w", encoding="utf-8") as f:
    f.write(latex_content)

print("Updated latex successfully.")
