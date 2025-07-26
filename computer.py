import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO

# 页面设置
st.title("S = ∑l_i + ∑(l_i * l_{i+1}) Distribution Analysis")

# 输入参数
n = st.number_input("输入 n（变量个数）", min_value=1, max_value=10000, value=3)
num_trials = st.number_input("模拟次数（采点数）", min_value=1000, max_value=100000000, value=10000, step=1000)
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
@st.cache_data
def estimate_s_values(n, num_trials, x_param, seed=42):
    l_values = generate_l_values(n, num_trials, x_param, seed)
    s_values = []
    
    for trial in range(num_trials):
        l = l_values[trial]
        # 计算 ∑l_i
        sum_l = np.sum(l)
        # 计算 ∑(l_i * l_{i+1})，注意 l_{n+1} = l_1
        l_shifted = np.roll(l, -1)
        sum_prod = np.sum(l * l_shifted)
        # 计算S
        S = sum_l + sum_prod
        s_values.append(S)
    
    return np.array(s_values), l_values

# 画图函数
def plot_s_distribution(values, bins, chart_type, n, num_trials, x_param):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if chart_type == "直方图（柱状图）":
        ax.hist(values, bins=bins, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Histogram')
    elif chart_type == "折线图（平滑点图）":
        counts, bin_edges = np.histogram(values, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, counts, color='blue', linewidth=2, label='Line')
    elif chart_type == "密度曲线（KDE）":
        sns.kdeplot(values, ax=ax, fill=True, color='purple', alpha=0.4, linewidth=2, label='KDE')

    ax.axvline(0, color='red', linestyle='--', label='S = 0')
    ax.set_title(f"S distribution (x = {x_param:.3f})\nn = {n}, num_trials = {num_trials}")
    ax.set_xlabel("S Value")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True)

    return fig

def plot_lavg_distribution(values, bins, chart_type, n, num_trials, x_param, threshold):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if chart_type == "直方图（柱状图）":
        ax.hist(values, bins=bins, density=True, color='lightgreen', edgecolor='black', alpha=0.7, label='Histogram')
    elif chart_type == "折线图（平滑点图）":
        counts, bin_edges = np.histogram(values, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, counts, color='green', linewidth=2, label='Line')
    elif chart_type == "密度曲线（KDE）":
        sns.kdeplot(values, ax=ax, fill=True, color='orange', alpha=0.4, linewidth=2, label='KDE')

    ax.axvline(threshold, color='red', linestyle='--', label=f'lavg = {threshold:.3f} (1+x)')
    ax.set_title(f"lavg distribution (x = {x_param:.3f}, S > 0)\nn = {n}, num_trials = {len(values)}")
    ax.set_xlabel("lavg Value")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True)

    return fig

# 初始化session state
if 'results_computed' not in st.session_state:
    st.session_state.results_computed = False
if 's_values' not in st.session_state:
    st.session_state.s_values = None
if 'l_values' not in st.session_state:
    st.session_state.l_values = None
if 's_positive' not in st.session_state:
    st.session_state.s_positive = None
if 'lavg_values' not in st.session_state:
    st.session_state.lavg_values = None
if 'fig_s' not in st.session_state:
    st.session_state.fig_s = None
if 'fig_lavg' not in st.session_state:
    st.session_state.fig_lavg = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = None
if 'p_s_gt_0' not in st.session_state:
    st.session_state.p_s_gt_0 = None
if 'p_s_lt_0' not in st.session_state:
    st.session_state.p_s_lt_0 = None
if 'mean_s_positive' not in st.session_state:
    st.session_state.mean_s_positive = None
if 'mean_s_negative' not in st.session_state:
    st.session_state.mean_s_negative = None
if 'p_lavg_gt_threshold' not in st.session_state:
    st.session_state.p_lavg_gt_threshold = None
if 'p_lavg_lt_threshold' not in st.session_state:
    st.session_state.p_lavg_lt_threshold = None
if 'mean_lavg_gt_threshold' not in st.session_state:
    st.session_state.mean_lavg_gt_threshold = None
if 'mean_lavg_lt_threshold' not in st.session_state:
    st.session_state.mean_lavg_lt_threshold = None
if 'num_trials' not in st.session_state:
    st.session_state.num_trials = None

# 显示按钮
if st.button("生成分析结果"):
    with st.spinner("正在模拟并计算..."):
        s_values, l_values = estimate_s_values(n, num_trials, x_param)
        
        # 保存到session state
        st.session_state.s_values = s_values
        st.session_state.l_values = l_values
        st.session_state.results_computed = True
        st.session_state.num_trials = num_trials

        # S的概率和平均值计算
        p_s_gt_0 = np.mean(s_values > 0)
        p_s_lt_0 = np.mean(s_values < 0)
        
        # S>0和S<0的平均值
        s_positive = s_values[s_values > 0]
        s_negative = s_values[s_values < 0]
        
        # 保存到session state
        st.session_state.s_positive = s_positive
        
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
            
            # 保存到session state
            st.session_state.lavg_values = lavg_values
            
            # lavg的统计分析
            threshold = 1 + x_param
            st.session_state.threshold = threshold
            
            p_lavg_gt_threshold = np.mean(lavg_values > threshold)
            p_lavg_lt_threshold = np.mean(lavg_values < threshold)
            
            # lavg>1+x和lavg<1+x的平均值
            lavg_gt_threshold = lavg_values[lavg_values > threshold]
            lavg_lt_threshold = lavg_values[lavg_values < threshold]
            
            mean_lavg_gt_threshold = np.mean(lavg_gt_threshold) if len(lavg_gt_threshold) > 0 else 0
            mean_lavg_lt_threshold = np.mean(lavg_lt_threshold) if len(lavg_lt_threshold) > 0 else 0
            
            # 保存统计结果到session state
            st.session_state.p_s_gt_0 = p_s_gt_0
            st.session_state.p_s_lt_0 = p_s_lt_0
            st.session_state.mean_s_positive = mean_s_positive
            st.session_state.mean_s_negative = mean_s_negative
            st.session_state.p_lavg_gt_threshold = p_lavg_gt_threshold
            st.session_state.p_lavg_lt_threshold = p_lavg_lt_threshold
            st.session_state.mean_lavg_gt_threshold = mean_lavg_gt_threshold
            st.session_state.mean_lavg_lt_threshold = mean_lavg_lt_threshold

        else:
            st.error("没有S > 0的样本，无法计算lavg")
            # 保存统计结果到session state
            st.session_state.p_s_gt_0 = p_s_gt_0
            st.session_state.p_s_lt_0 = p_s_lt_0
            st.session_state.mean_s_positive = mean_s_positive
            st.session_state.mean_s_negative = mean_s_negative

# 如果已经计算过结果，显示结果和下载区域
if st.session_state.results_computed:
    # 显示S的统计结果
    st.markdown("### 📊 S的统计结果")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**S的概率**")
        st.write(f"**P(S > 0)** = `{st.session_state.p_s_gt_0:.6f}`")
        st.write(f"**P(S < 0)** = `{st.session_state.p_s_lt_0:.6f}`")
    
    with col2:
        st.markdown("**S的平均值**")
        st.write(f"**E[S | S > 0]** = `{st.session_state.mean_s_positive:.6f}`")
        st.write(f"**E[S | S < 0]** = `{st.session_state.mean_s_negative:.6f}`")

    # 生成并显示S分布图
    st.markdown("### 📈 S分布图")
    s_values = st.session_state.s_values
    fig_s = plot_s_distribution(s_values, bins, chart_type, n, num_trials, x_param)
    st.pyplot(fig_s)
    st.session_state.fig_s = fig_s

    # 显示lavg的统计结果和图表
    if st.session_state.s_positive is not None and len(st.session_state.s_positive) > 0:
        lavg_values = st.session_state.lavg_values
        threshold = st.session_state.threshold
        
        st.markdown("### 📊 lavg的统计结果（当S > 0时）")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**lavg的概率**")
            st.write(f"**P(lavg > 0 | S > 0)** = `{1.0:.6f}`")
            st.write(f"**P(lavg > {threshold:.3f} | S > 0)** = `{st.session_state.p_lavg_gt_threshold:.6f}`")
            st.write(f"**P(lavg < {threshold:.3f} | S > 0)** = `{st.session_state.p_lavg_lt_threshold:.6f}`")
        
        with col4:
            st.markdown("**lavg的平均值**")
            st.write(f"**E[lavg | S > 0]** = `{np.mean(lavg_values):.6f}`")
            st.write(f"**E[lavg | lavg > {threshold:.3f}]** = `{st.session_state.mean_lavg_gt_threshold:.6f}`")
            st.write(f"**E[lavg | lavg < {threshold:.3f}]** = `{st.session_state.mean_lavg_lt_threshold:.6f}`")

        # 生成并显示lavg分布图
        st.markdown("### 📈 lavg分布图（当S > 0时）")
        fig_lavg = plot_lavg_distribution(lavg_values, bins, chart_type, n, num_trials, x_param, threshold)
        st.pyplot(fig_lavg)
        st.session_state.fig_lavg = fig_lavg

    # 显示下载区域
    st.markdown("---")
    st.markdown("### 📥 下载结果")
    
    # 创建下载列
    col_download1, col_download2, col_download3, col_download4 = st.columns(4)
    
    with col_download1:
        if st.session_state.fig_s is not None:
            buf_s = BytesIO()
            st.session_state.fig_s.savefig(buf_s, format="png", dpi=1200, bbox_inches="tight")
            st.download_button(
                label="下载S分布图（PNG）",
                data=buf_s.getvalue(),
                file_name=f"s_distribution_n{n}_x{x_param:.3f}_{chart_type}.png",
                mime="image/png"
            )
    
    with col_download2:
        if st.session_state.s_values is not None and st.session_state.l_values is not None:
            # 导出每次采样的S值和对应的l值
            sample_data = []
            for trial in range(st.session_state.num_trials):
                l_trial = st.session_state.l_values[trial]
                s_trial = st.session_state.s_values[trial]
                row = {'trial': trial + 1, 'S_value': s_trial}
                # 添加每个l_i的值
                for i in range(n):
                    row[f'l_{i+1}'] = l_trial[i]
                sample_data.append(row)
            
            sample_df = pd.DataFrame(sample_data)
            csv_sample = sample_df.to_csv(index=False)
            st.download_button(
                label="下载每次采样数据（CSV）",
                data=csv_sample,
                file_name=f"sample_data_n{n}_x{x_param:.3f}.csv",
                mime="text/csv"
            )
    
    if st.session_state.s_positive is not None and len(st.session_state.s_positive) > 0:
        with col_download3:
            if st.session_state.fig_lavg is not None:
                buf_lavg = BytesIO()
                st.session_state.fig_lavg.savefig(buf_lavg, format="png", dpi=1200, bbox_inches="tight")
                st.download_button(
                    label="下载lavg分布图（PNG）",
                    data=buf_lavg.getvalue(),
                    file_name=f"lavg_distribution_n{n}_x{x_param:.3f}_{chart_type}.png",
                    mime="image/png"
                )
        
        with col_download4:
            if st.session_state.lavg_values is not None:
                # 导出S>0对应的lavg值
                lavg_sample_data = []
                for i, s_val in enumerate(st.session_state.s_positive):
                    lavg_val = st.session_state.lavg_values[i]
                    row = {
                        'trial_index': i + 1,  # 在S>0样本中的索引
                        'S_positive_value': s_val,
                        'lavg_value': lavg_val,
                        'lavg_gt_threshold': lavg_val > st.session_state.threshold
                    }
                    lavg_sample_data.append(row)
                
                lavg_sample_df = pd.DataFrame(lavg_sample_data)
                csv_lavg_sample = lavg_sample_df.to_csv(index=False)
                st.download_button(
                    label="下载lavg采样数据（CSV）",
                    data=csv_lavg_sample,
                    file_name=f"lavg_sample_data_n{n}_x{x_param:.3f}.csv",
                    mime="text/csv"
                )

# 显示公式说明
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
