import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，避免显示问题
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO

# 设置matplotlib样式
plt.style.use('default')
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = 'white'

# 设置字体大小
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 12

# 页面设置
st.title("S = ∑l_i + ∑(l_i * l_{i+1}) Distribution Analysis")

# 输入参数
n = st.number_input("输入 n（变量个数）", min_value=1, max_value=10000, value=3)
num_trials = st.number_input("模拟次数（采点数）", min_value=1000, max_value=10000000, value=10000, step=1000)

# 添加采样次数警告
if num_trials > 1000000:
    st.warning("⚠️ 采样次数超过100万，可能导致性能问题。建议使用较小的采样次数。")
elif num_trials > 500000:
    st.info("ℹ️ 采样次数较大，计算可能需要较长时间，请耐心等待。")

bins = st.slider("分桶数（bins）", min_value=10, max_value=1000, value=100)

# 参数x设置
st.markdown("#### 参数x设置")
x_param = st.number_input(
    "输入参数x的值", 
    min_value=0.0, 
    value=1.0
)
st.write(f"当前使用: x = {x_param:.6f}")

# 图表类型选择
chart_type = st.selectbox("图表类型", ["直方图（柱状图）", "密度曲线（KDE）", "折线图（平滑点图）"])

# 生成随机变量的函数
def generate_l_values(n, num_trials, x_param, seed=42):
    np.random.seed(seed)
    # B) l_i 等概率分布在[-1-x,1+x]
    return np.random.uniform(-1-x_param, 1+x_param, (num_trials, n))

# 计算S值的函数
def estimate_s_values(n, num_trials, x_param, seed=42):
    l_values = generate_l_values(n, num_trials, x_param, seed)
    
    # 添加进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 使用向量化操作计算S值
    # 计算 ∑l_i (对每行求和)
    sum_l = np.sum(l_values, axis=1)
    
    # 计算 ∑(l_i * l_{i+1})，注意 l_{n+1} = l_1
    l_shifted = np.roll(l_values, -1, axis=1)
    sum_prod = np.sum(l_values * l_shifted, axis=1)
    
    # 计算S
    s_values = sum_l + sum_prod
    
    # 更新进度条到100%
    progress_bar.progress(1.0)
    status_text.text(f"计算完成: {num_trials:,} 次采样")
    
    # 清除进度条
    progress_bar.empty()
    status_text.empty()
    
    return s_values, l_values

# 画图函数
def plot_s_distribution(values, bins, chart_type, n, num_trials, x_param):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')
    
    if chart_type == "直方图（柱状图）":
        ax.hist(values, bins=bins, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Histogram')
    elif chart_type == "折线图（平滑点图）":
        counts, bin_edges = np.histogram(values, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, counts, color='blue', linewidth=3.0, label='Line')
    elif chart_type == "密度曲线（KDE）":
        sns.kdeplot(values, ax=ax, fill=True, color='purple', alpha=0.4, linewidth=2, label='KDE')

    ax.axvline(0, color='red', linestyle='--', linewidth=3.0, label='S = 0')
    ax.set_title(f"S distribution (x = {x_param:.3f})\nn = {n}, num_trials = {num_trials}", fontsize=16, pad=20)
    ax.set_xlabel("S value", fontsize=32)
    ax.set_ylabel("Probability density", fontsize=32)
    
    # 设置刻度标签字体大小和刻度线粗细
    ax.tick_params(axis='both', which='major', labelsize=28, width=3.0, length=10)

    fig.patch.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3.0) # 设置边框宽度
    return fig

def plot_lavg_distribution(values, bins, chart_type, n, num_trials, x_param, threshold):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')
    
    if chart_type == "直方图（柱状图）":
        ax.hist(values, bins=bins, density=True, color='lightgreen', edgecolor='black', alpha=0.7, label='Histogram')
    elif chart_type == "折线图（平滑点图）":
        counts, bin_edges = np.histogram(values, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, counts, color='green', linewidth=3.0, label='Line')
    elif chart_type == "密度曲线（KDE）":
        sns.kdeplot(values, ax=ax, fill=True, color='orange', alpha=0.4, linewidth=2, label='KDE')

    ax.axvline(threshold, color='red', linestyle='--', linewidth=3.0, label=f'lavg = {threshold:.3f} (1+x)')
    ax.set_title(f"lavg distribution (x = {x_param:.3f}, S > 0)\nn = {n}, num_trials = {len(values)}", fontsize=16, pad=20)
    ax.set_xlabel("lavg value", fontsize=32)
    ax.set_ylabel("Probability density", fontsize=32)
    
    # 设置刻度标签字体大小和刻度线粗细
    ax.tick_params(axis='both', which='major', labelsize=28, width=3.0, length=10)

    # 确保图形背景是白色
    fig.patch.set_facecolor('white')
    # 添加黑色边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3.0)
    return fig

# 显示按钮
if st.button("生成分析结果"):
    with st.spinner("正在模拟并计算..."):
        s_values, l_values = estimate_s_values(n, num_trials, x_param)
        
        # S的概率和平均值计算
        p_s_gt_0 = np.mean(s_values > 0)
        p_s_lt_0 = np.mean(s_values < 0)
        
        # S>0和S<0的平均值
        s_positive = s_values[s_values > 0]
        s_negative = s_values[s_values < 0]
        
        mean_s_positive = np.mean(s_positive) if len(s_positive) > 0 else 0
        mean_s_negative = np.mean(s_negative) if len(s_negative) > 0 else 0

        # 计算lavg（只对S > 0的部分）
        if len(s_positive) > 0:
            # lavg = √(1 + (4(sin(π/n))²)/n * S)，其中S > 0
            sin_term = np.sin(np.pi / n)
            sin_squared = sin_term**2
            coefficient = (4 * sin_squared) / n
            
            # 计算根号内的值（只对S > 0）
            sqrt_argument = 1 + coefficient * s_positive
            
            # 计算lavg（只对S > 0的部分）
            lavg_values = np.sqrt(sqrt_argument)
            
            # lavg的统计分析
            threshold = 1 + x_param
            
            p_lavg_gt_threshold = np.mean(lavg_values > threshold)
            p_lavg_lt_threshold = np.mean(lavg_values < threshold)
            
            # lavg>1+x和lavg<1+x的平均值
            lavg_gt_threshold = lavg_values[lavg_values > threshold]
            lavg_lt_threshold = lavg_values[lavg_values < threshold]
            
            mean_lavg_gt_threshold = np.mean(lavg_gt_threshold) if len(lavg_gt_threshold) > 0 else 0
            mean_lavg_lt_threshold = np.mean(lavg_lt_threshold) if len(lavg_lt_threshold) > 0 else 0

        else:
            st.error("没有S > 0的样本，无法计算lavg")
            lavg_values = None
            threshold = None
            p_lavg_gt_threshold = None
            p_lavg_lt_threshold = None
            mean_lavg_gt_threshold = None
            mean_lavg_lt_threshold = None

        # 显示结果
        st.markdown("### 📊 S的统计结果")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**S的概率**")
            st.write(f"**P(S > 0)** = `{p_s_gt_0:.6f}`")
            st.write(f"**P(S < 0)** = `{p_s_lt_0:.6f}`")
        
        with col2:
            st.markdown("**S的平均值**")
            st.write(f"**E[S | S > 0]** = `{mean_s_positive:.6f}`")
            st.write(f"**E[S | S < 0]** = `{mean_s_negative:.6f}`")

        st.markdown("### 📈 S分布图")
        fig_s = plot_s_distribution(s_values, bins, chart_type, n, num_trials, x_param)
        st.pyplot(fig_s)
        plt.close(fig_s)

        if s_positive is not None and len(s_positive) > 0:
            st.markdown("### 📊 lavg的统计结果（当S > 0时）")
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("**lavg的概率**")
                st.write(f"**P(lavg > 0 | S > 0)** = `{1.0:.6f}`")
                st.write(f"**P(lavg > {threshold:.3f} | S > 0)** = `{p_lavg_gt_threshold:.6f}`")
                st.write(f"**P(lavg < {threshold:.3f} | S > 0)** = `{p_lavg_lt_threshold:.6f}`")
            
            with col4:
                st.markdown("**lavg的平均值**")
                st.write(f"**E[lavg | S > 0]** = `{np.mean(lavg_values):.6f}`")
                st.write(f"**E[lavg | lavg > {threshold:.3f}]** = `{mean_lavg_gt_threshold:.6f}`")
                st.write(f"**E[lavg | lavg < {threshold:.3f}]** = `{mean_lavg_lt_threshold:.6f}`")

            # 生成并显示lavg分布图
            st.markdown("### 📈 lavg分布图（当S > 0时）")
            fig_lavg = plot_lavg_distribution(lavg_values, bins, chart_type, n, num_trials, x_param, threshold)
            st.pyplot(fig_lavg)
            plt.close(fig_lavg)

            st.markdown("---")
            st.markdown("### 📥 下载图片")
            
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                try:
                    download_fig_s = plot_s_distribution(s_values, bins, chart_type, n, num_trials, x_param)
                    buf_s = BytesIO()
                    download_fig_s.savefig(buf_s, format="png", dpi=300, bbox_inches="tight", 
                                         facecolor='white', edgecolor='none')
                    buf_s.seek(0)
                    
                    st.download_button(
                        label="下载S分布图（PNG）",
                        data=buf_s.getvalue(),
                        file_name=f"s_distribution_n{n}_x{x_param:.3f}_{chart_type}.png",
                        mime="image/png",
                        key="download_s_fig"
                    )
                    plt.close(download_fig_s)
                except Exception as e:
                    st.error(f"下载S分布图时出错：{str(e)}")
            
            with col_download2:
                try:
                    download_fig_lavg = plot_lavg_distribution(lavg_values, bins, chart_type, n, num_trials, x_param, threshold)
                    buf_lavg = BytesIO()
                    download_fig_lavg.savefig(buf_lavg, format="png", dpi=300, bbox_inches="tight",
                                            facecolor='white', edgecolor='none')
                    buf_lavg.seek(0)
                    
                    st.download_button(
                        label="下载lavg分布图（PNG）",
                        data=buf_lavg.getvalue(),
                        file_name=f"lavg_distribution_n{n}_x{x_param:.3f}_{chart_type}.png",
                        mime="image/png",
                        key="download_lavg_fig"
                    )
                    plt.close(download_fig_lavg)
                except Exception as e:
                    st.error(f"下载lavg分布图时出错：{str(e)}")

st.markdown("---")
st.markdown("### 📝 公式说明")
st.latex(r"S = \sum_{i=1}^{n} l_i + \sum_{i=1}^{n} (l_i \cdot l_{i+1})")
st.markdown("其中 $l_{n+1} = l_1$（循环边界条件）")

st.latex(r"l_{avg} = \sqrt{1 + \frac{4\sin^2(\pi/n)}{n} \cdot S}")
st.markdown("其中 $S > 0$")

st.markdown("### 📋 分布条件说明")
st.markdown(f"""
**B)** $l_i$ 等概率分布在 $[-1-x, 1+x]$，概率密度 $p_i = \\frac{{0.5}}{{1+x}}$，其中 $x = {x_param:.3f}$
""")
