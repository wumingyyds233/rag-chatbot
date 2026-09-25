import streamlit as st

# 页面配置（必须是第一个 Streamlit 命令）
st.set_page_config(
    page_title="我的第一个网页",
    page_icon="🎈",
    layout="wide"
)

# 标题和文本
st.title("🎈 你好，Streamlit！")
st.header("这是一个二级标题")
st.subheader("这是一个三级标题")
st.write("这是一段普通文本，支持 **Markdown** 语法。")

# 输入组件
name = st.text_input("请输入你的名字：")
if name:
    st.success(f"你好，{name}！")

# 按钮
if st.button("点我"):
    st.balloons()   # 放气球动画

# 侧边栏
with st.sidebar:
    st.header("侧边栏")
    option = st.selectbox("选择一个选项", ["A", "B", "C"])
    st.write(f"你选了：{option}")

# 分栏
col1, col2 = st.columns(2)
with col1:
    st.metric("用户数", "1,234", "+12%")
with col2:
    st.metric("销售额", "¥56,789", "-3%")

# 数据展示
import pandas as pd
df = pd.DataFrame({
    "姓名": ["张三", "李四", "王五"],
    "年龄": [25, 30, 28],
    "城市": ["北京", "上海", "广州"]
})
st.dataframe(df)