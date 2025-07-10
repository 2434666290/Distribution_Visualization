import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# 页面设置
st.title("S = sum(x_i) + sum(x_i * x_{i+1}) Distribution Visualization")

# 输入参数
n = st.number_input("输入 n（变量个数）", min_value=1, max_value=1000, value=3)
num_trials = st.number_input("模拟次数（num_trials）", min_value=1000, max_value=1000000, value=10000, step=1000)
bins = st.slider("分桶数（bins）", min_value=10, max_value=1000, value=100)

# 图表类型选择
chart_type = st.selectbox("图表类型", ["直方图（柱状图）", "密度曲线（KDE）", "折线图（平滑点图）"])

# 模拟函数
@st.cache_data
def estimate_s_values(n, num_trials, seed=42):
    np.random.seed(seed)
    s_values = []
    for _ in range(num_trials):
        x = np.random.uniform(-1, 1, n)
        sum_x = np.sum(x)
        prod_x = np.sum(x * np.roll(x, -1))
        S = sum_x + prod_x
        s_values.append(S)
    return np.array(s_values)

# 画图函数
def plot_distribution(s_values, bins, chart_type):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if chart_type == "直方图（柱状图）":
        ax.hist(s_values, bins=bins, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Histogram')
    elif chart_type == "折线图（平滑点图）":
        counts, bin_edges = np.histogram(s_values, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, counts, color='blue', linewidth=2, label='Line')
    elif chart_type == "密度曲线（KDE）":
        sns.kdeplot(s_values, ax=ax, fill=True, color='purple', alpha=0.4, linewidth=2, label='KDE')

    ax.axvline(0, color='red', linestyle='--', label='S = 0')
    ax.set_title(f"Distribution Plot: S Value (n = {n}, trials = {num_trials})")
    ax.set_xlabel("S Value")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True)

    return fig

# 显示按钮
if st.button("生成图像并估计概率"):
    with st.spinner("正在模拟并绘制图像..."):
        s_values = estimate_s_values(n, num_trials)

        # 概率估计
        p_gt_0 = np.mean(s_values > 0)
        p_lt_0 = np.mean(s_values < 0)

        # 显示概率
        st.markdown(f"### 📊 概率估计结果")
        st.write(f"**P(S > 0)** ≈ `{p_gt_0:.5f}`")
        st.write(f"**P(S < 0)** ≈ `{p_lt_0:.5f}`")

        # 绘图
        fig = plot_distribution(s_values, bins, chart_type)
        st.pyplot(fig)

        # 下载图像
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label="下载图像（PNG）",
            data=buf.getvalue(),
            file_name=f"s_distribution_n{n}_{chart_type}.png",
            mime="image/png"
        )
