import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f0c29, #302b63, #24243e) !important;
    }
    [data-testid="stSidebar"] * { color: #e0e0ff !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 1rem; font-weight: 600; }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e3a, #2d2d5e);
        border: 1px solid rgba(130,100,255,0.3);
        border-radius: 14px;
        padding: 14px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    div[data-testid="metric-container"] label {
        color: #a0a8d4 !important; font-size: 0.78rem;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f0f0ff !important; font-size: 1.5rem; font-weight: 700;
    }

    h1 { color: #c9b8ff !important; font-weight: 800; }
    h2, h3 { color: #a89fe8 !important; font-weight: 700; }
    .stApp { background: #0d0d1a; }

    .card {
        background: linear-gradient(135deg, #12122a, #1e1e40);
        border: 1px solid rgba(130,100,255,0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .dev-card {
        background: linear-gradient(135deg, #1a0533, #2e0a5e, #1a1a4e);
        border: 2px solid rgba(180,100,255,0.4);
        border-radius: 20px;
        padding: 36px;
        text-align: center;
        box-shadow: 0 8px 40px rgba(140,60,255,0.3);
        margin-bottom: 24px;
    }
    .dev-card h1 { font-size: 2.4rem !important; color: #d4aaff !important; }
    .dev-card p  { color: #c0b8e0; font-size: 1.05rem; }
    .badge {
        display: inline-block;
        background: rgba(130,60,255,0.25);
        border: 1px solid rgba(170,100,255,0.5);
        border-radius: 30px;
        padding: 6px 20px;
        font-size: 0.9rem;
        color: #d0baff;
        margin: 4px;
    }
    .contact-row {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 18px;
        flex-wrap: wrap;
    }
    .contact-item {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px 22px;
        font-size: 0.95rem;
        color: #e8e0ff;
    }
    .hero-bg {
        background: linear-gradient(rgba(13,13,26,0.72), rgba(13,13,26,0.88)),
                    url('https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=1400&q=80')
                    center/cover no-repeat;
        border-radius: 20px;
        padding: 60px 40px;
        margin-bottom: 30px;
    }
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(130,100,255,0.5), transparent);
        margin: 28px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 12px 24px !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
        box-shadow: 0 4px 20px rgba(109,40,217,0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding='latin1')
    df = df.drop_duplicates()
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=False)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=False)
    df['Year']          = df['Order Date'].dt.year
    df['Month']         = df['Order Date'].dt.month
    df['Month Name']    = df['Order Date'].dt.strftime('%b')
    df['Quarter']       = df['Order Date'].dt.quarter.map({1:'Q1',2:'Q2',3:'Q3',4:'Q4'})
    df['Profit Margin'] = (df['Profit'] / df['Sales'].replace(0, np.nan)) * 100
    df['Delivery Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['Profit/Loss']   = df['Profit'].apply(lambda x: 'Profit' if x >= 0 else 'Loss')
    return df


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Sales Analytics")
    st.markdown("---")
    uploaded = st.file_uploader("📁 Upload Superstore CSV", type=["csv"])
    st.markdown("---")
    nav = st.radio(
        "🗺️ Navigation",
        ["🏠 Project Overview",
         "📈 Trend Analysis",
         "🤖 ML Sales Prediction"]
    )
    st.markdown("---")
    st.markdown("<small style='color:#8880cc'>© 2024 Deepak Baskar</small>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────
df = None
if uploaded is not None:
    try:
        df = load_data(uploaded)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")


# ═════════════════════════════════════════════════════════════
#  PAGE 1 — PROJECT OVERVIEW
# ═════════════════════════════════════════════════════════════
if nav == "🏠 Project Overview":

    st.markdown("""
    <div class="hero-bg">
        <h1 style="font-size:2.8rem;color:#d4aaff;margin-bottom:10px;">
            📊 Sales Analytics Dashboard
        </h1>
        <p style="color:#c8c0f0;font-size:1.15rem;max-width:720px;line-height:1.8;">
            An end-to-end interactive <strong>Business Intelligence Platform</strong>
            built on the <em>Sample Superstore</em> dataset — combining deep exploratory
            analysis, real-time multi-filter exploration, and ML-powered sales forecasting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dev-card">
        <div style="font-size:4.5rem;margin-bottom:12px;">👨‍💻</div>
        <h1>Deepak Baskar</h1>
        <p style="font-size:1.1rem;color:#b8a8e8;margin-bottom:14px;">
            Data Analyst 
        </p>
        <div>
            <span class="badge">🐍 Python</span>
            <span class="badge">📊 Streamlit</span>
            <span class="badge">🤖 Scikit-Learn</span>
            <span class="badge">📈 Plotly</span>
            <span class="badge">🧮 Pandas</span>
            <span class="badge">🌲 Random Forest</span>
        </div>
        <div class="contact-row">
            <div class="contact-item">📱 +91 9965463281</div>
            <div class="contact-item">✉️ baskardeepak27@gmail.com</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🎯 Project Name</h3>
            <p style="color:#c8c0f0;font-size:1.05rem;">
                <strong>Sales Analytics &amp; Forecasting Dashboard</strong>
            </p>
            <hr class='section-divider'>
            <h3>📋 Project Description</h3>
            <p style="color:#b0a8d8;line-height:1.8;">
                This dashboard provides comprehensive analysis of retail sales data from the
                <strong>Sample Superstore</strong> dataset. Business users can explore
                performance across segments, regions, categories, and time periods —
                and predict future sales using trained ML models.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>🛠️ Key Features</h3>
            <ul style="color:#b0a8d8;line-height:2.2;">
                <li>📊 Profit / Loss / Sales KPI cards</li>
                <li>📈 Yearly &amp; Monthly trend analysis</li>
                <li>🗂️ Category &amp; Sub-Category breakdowns</li>
                <li>🚚 Ship Mode &amp; Segment analysis</li>
                <li>🌎 Region-wise profit heatmaps</li>
                <li>🤖 Random Forest ML Sales Prediction</li>
                <li>📉 Discount impact visualizations</li>
                <li>🚀 Cumulative growth area charts</li>
                <li>📦 Delivery days analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>📦 Dataset — Sample Superstore</h3>
        <p style="color:#b0a8d8;line-height:1.8;">
            The dataset covers <strong>retail orders across the United States</strong>,
            containing columns like Order Date, Ship Date, Ship Mode, Segment, Region,
            Category, Sub-Category, Product Name, Sales, Quantity, Discount, and Profit.
            It spans multiple years and serves as a benchmark dataset for BI and analytics.
        </p>
        <div style="margin-top:14px;">
            <span class="badge">📅 Multi-Year</span>
            <span class="badge">🌎 4 US Regions</span>
            <span class="badge">🗂️ 3 Categories</span>
            <span class="badge">📦 17 Sub-Categories</span>
            <span class="badge">👥 3 Segments</span>
            <span class="badge">🚚 4 Ship Modes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE 2 — TREND ANALYSIS
# ═════════════════════════════════════════════════════════════
elif nav == "📈 Trend Analysis":
    st.title("📈 Trend Analysis")

    if df is None:
        st.warning("⬆️ Please upload your **Superstore CSV** using the sidebar.")
        st.stop()

    # ── GLOBAL KPIs ──────────────────────────────────────────
    st.markdown("### 📊 Overall KPIs")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("💰 Total Sales",    f"${df['Sales'].sum():,.0f}")
    k2.metric("📈 Total Profit",   f"${df['Profit'].sum():,.0f}")
    k3.metric("📦 Total Qty",      f"{df['Quantity'].sum():,}")
    k4.metric("🏷️ Avg Discount",   f"{df['Discount'].mean()*100:.1f}%")
    k5.metric("✅ Profit Orders",  f"{len(df[df['Profit']>=0]):,}")
    k6.metric("❌ Loss Orders",    f"{len(df[df['Profit']<0]):,}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Row 1
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### 📅 Monthly Sales by Year")
        monthly = (df.groupby(['Year','Month','Month Name'])['Sales']
                   .sum().reset_index().sort_values(['Year','Month']))
        fig = px.line(monthly, x='Month Name', y='Sales', color='Year',
                      markers=True, template='plotly_dark',
                      color_discrete_sequence=px.colors.qualitative.Vivid,
                      category_orders={'Month Name':
                          ['Jan','Feb','Mar','Apr','May','Jun',
                           'Jul','Aug','Sep','Oct','Nov','Dec']})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown("#### 📆 Quarterly Profit")
        qtr = df.groupby(['Year','Quarter'])['Profit'].sum().reset_index()
        qtr['YQ'] = qtr['Year'].astype(str) + ' ' + qtr['Quarter']
        fig2 = px.bar(qtr, x='YQ', y='Profit', color='Quarter',
                      template='plotly_dark',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2
    t3, t4 = st.columns(2)
    with t3:
        st.markdown("#### 🏷️ Discount vs Profit Scatter")
        sample = df.sample(min(2000, len(df)), random_state=1)
        fig3 = px.scatter(sample, x='Discount', y='Profit',
                          color='Category', size='Sales',
                          template='plotly_dark',
                          color_discrete_sequence=px.colors.qualitative.Bold,
                          hover_data=['Sub-Category','Segment'])
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    with t4:
        st.markdown("#### 🚀 Cumulative Sales Growth")
        cum = df.sort_values('Order Date').copy()
        cum['Cumulative Sales'] = cum['Sales'].cumsum()
        fig4 = px.area(cum, x='Order Date', y='Cumulative Sales',
                       template='plotly_dark',
                       color_discrete_sequence=['#8b5cf6'])
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3
    t5, t6 = st.columns(2)
    with t5:
        st.markdown("#### 🚚 Ship Mode Sales Over Years")
        ship_yr = df.groupby(['Year','Ship Mode'])['Sales'].sum().reset_index()
        fig5 = px.line(ship_yr, x='Year', y='Sales', color='Ship Mode',
                       markers=True, template='plotly_dark',
                       color_discrete_sequence=px.colors.qualitative.Vivid)
        fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig5, use_container_width=True)

    with t6:
        st.markdown("#### 👥 Segment Profit Over Years")
        seg_yr = df.groupby(['Year','Segment'])['Profit'].sum().reset_index()
        fig6 = px.bar(seg_yr, x='Year', y='Profit', color='Segment',
                      barmode='group', template='plotly_dark',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig6.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig6, use_container_width=True)

    # Row 4
    t7, t8 = st.columns(2)
    with t7:
        st.markdown("#### 🌎 Category × Region Profit Heatmap")
        if 'Region' in df.columns:
            pivot = df.pivot_table(values='Profit', index='Region',
                                   columns='Category', aggfunc='sum')
            fig7 = px.imshow(pivot, text_auto='.0f', aspect='auto',
                             color_continuous_scale='RdYlGn',
                             template='plotly_dark')
            fig7.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("Region column not found.")

    with t8:
        st.markdown("#### 📦 Top 10 Sub-Categories by Sales")
        top_sub = (df.groupby('Sub-Category')['Sales']
                   .sum().sort_values(ascending=False)
                   .head(10).reset_index())
        fig8 = px.bar(top_sub, x='Sales', y='Sub-Category', orientation='h',
                      color='Sales', color_continuous_scale='Purples',
                      template='plotly_dark')
        fig8.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig8, use_container_width=True)

    # Row 5
    t9, t10 = st.columns(2)
    with t9:
        st.markdown("#### 📂 Category-wise Sales")
        cat_s = df.groupby('Category')['Sales'].sum().reset_index()
        fig9 = px.bar(cat_s, x='Category', y='Sales', color='Category',
                      color_discrete_sequence=['#7c3aed','#4f46e5','#0ea5e9'],
                      template='plotly_dark', text_auto='.2s')
        fig9.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig9, use_container_width=True)

    with t10:
        st.markdown("#### 🟢 Profit vs Loss Distribution")
        pl = df['Profit/Loss'].value_counts().reset_index()
        pl.columns = ['Status', 'Count']
        fig10 = px.pie(pl, names='Status', values='Count', color='Status',
                       color_discrete_map={'Profit':'#4ade80','Loss':'#f87171'},
                       template='plotly_dark', hole=0.45)
        fig10.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig10, use_container_width=True)

    # Row 6
    t11, t12 = st.columns(2)
    with t11:
        st.markdown("#### 🏷️ Sub-Category Profit / Loss")
        sub_p = df.groupby('Sub-Category')['Profit'].sum().sort_values().reset_index()
        sub_p['Color'] = sub_p['Profit'].apply(
            lambda x: '#4ade80' if x >= 0 else '#f87171')
        fig11 = px.bar(sub_p, x='Profit', y='Sub-Category', orientation='h',
                       color='Color', color_discrete_map='identity',
                       template='plotly_dark')
        fig11.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig11, use_container_width=True)

    with t12:
        st.markdown("#### 🚢 Ship Mode × Segment Heatmap")
        hm = df.pivot_table(values='Sales', index='Segment',
                            columns='Ship Mode', aggfunc='sum', fill_value=0)
        fig12 = px.imshow(hm, text_auto='.2s', aspect='auto',
                          color_continuous_scale='Purples',
                          template='plotly_dark')
        fig12.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig12, use_container_width=True)

    # Full-width charts
    st.markdown("#### 📊 Yearly Sales & Profit Comparison")
    yr_both = df.groupby('Year')[['Sales','Profit']].sum().reset_index()
    fig_yr = go.Figure()
    fig_yr.add_trace(go.Bar(x=yr_both['Year'], y=yr_both['Sales'],
                            name='Sales', marker_color='#7c3aed'))
    fig_yr.add_trace(go.Bar(x=yr_both['Year'], y=yr_both['Profit'],
                            name='Profit', marker_color='#4ade80'))
    fig_yr.add_trace(go.Scatter(x=yr_both['Year'], y=yr_both['Sales'],
                                mode='lines+markers', name='Sales Trend',
                                line=dict(color='#a78bfa', width=2, dash='dot')))
    fig_yr.update_layout(barmode='group', template='plotly_dark',
                         paper_bgcolor='rgba(0,0,0,0)',
                         plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_yr, use_container_width=True)

    st.markdown("#### 🚚 Delivery Days Distribution by Ship Mode")
    fig_del = px.box(df, x='Ship Mode', y='Delivery Days', color='Ship Mode',
                     template='plotly_dark',
                     color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_del.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig_del, use_container_width=True)

    st.markdown("#### 👥 Segment — Sales & Profit")
    seg_all = df.groupby('Segment')[['Sales','Profit']].sum().reset_index()
    fig_seg = px.bar(seg_all, x='Segment', y=['Sales','Profit'],
                     barmode='group',
                     color_discrete_sequence=['#8b5cf6','#4ade80'],
                     template='plotly_dark')
    fig_seg.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_seg, use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  PAGE 3 — ML SALES PREDICTION
# ═════════════════════════════════════════════════════════════
elif nav == "🤖 ML Sales Prediction":
    st.title("🤖 ML Sales Prediction")
    st.markdown(
        "<p style='color:#a0a8d4;font-size:1.05rem;'>Predict future sales using a "
        "<strong>Random Forest Regressor</strong> trained on your Superstore data.</p>",
        unsafe_allow_html=True)

    if df is None:
        st.warning("⬆️ Please upload your **Superstore CSV** using the sidebar.")
        st.stop()

    @st.cache_resource
    def train_model(n_rows):
        mdf = df.copy()
        base_cats = ['Category', 'Sub-Category', 'Segment', 'Ship Mode']
        if 'Region' in mdf.columns:
            base_cats.append('Region')
        cat_cols = [c for c in base_cats if c in mdf.columns]

        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            mdf[col + '_enc'] = le.fit_transform(mdf[col].astype(str))
            encoders[col] = le

        feature_cols = [c + '_enc' for c in cat_cols] + \
                       ['Quantity', 'Discount', 'Year', 'Month']
        feature_cols = [c for c in feature_cols if c in mdf.columns]

        X = mdf[feature_cols].dropna()
        y = mdf.loc[X.index, 'Sales']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        return rf, encoders, feature_cols, cat_cols, mae, r2, \
               X_test, y_test.values, y_pred

    with st.spinner("🌲 Training Random Forest model… please wait"):
        model, encoders, feature_cols, cat_cols, mae, r2, \
        X_test, y_test_vals, y_pred_vals = train_model(len(df))

    # Metrics
    st.markdown("### 📊 Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 R² Score",       f"{r2:.4f}")
    m2.metric("📉 Mean Abs Error", f"${mae:,.2f}")
    m3.metric("🌲 Trees",          "150")
    m4.metric("📦 Training Rows",  f"{int(len(df)*0.8):,}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Actual vs Predicted
    st.markdown("#### 🔵 Actual vs Predicted Sales — Test Set (first 200 samples)")
    avp = pd.DataFrame({
        'Index':     list(range(200)),
        'Actual':    y_test_vals[:200],
        'Predicted': y_pred_vals[:200]
    })
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(x=avp['Index'], y=avp['Actual'],
                                 mode='lines', name='Actual',
                                 line=dict(color='#4ade80', width=2)))
    fig_avp.add_trace(go.Scatter(x=avp['Index'], y=avp['Predicted'],
                                 mode='lines', name='Predicted',
                                 line=dict(color='#f87171', width=2, dash='dash')))
    fig_avp.update_layout(template='plotly_dark',
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig_avp, use_container_width=True)

    c_res1, c_res2 = st.columns(2)
    with c_res1:
        st.markdown("#### 📊 Residual Distribution")
        residuals = y_test_vals[:200] - y_pred_vals[:200]
        fig_res = px.histogram(residuals, nbins=40,
                               color_discrete_sequence=['#8b5cf6'],
                               template='plotly_dark',
                               labels={'value': 'Residual (Actual − Predicted)'})
        fig_res.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_res, use_container_width=True)

    with c_res2:
        st.markdown("#### 🌟 Feature Importance")
        fi = pd.DataFrame({
            'Feature':    feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Purples',
                        template='plotly_dark')
        fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### 🔮 Predict Sales for a New Order")
    st.markdown(
        "<p style='color:#a0a8d4;'>Fill in the order details and click Predict.</p>",
        unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    p4, p5, p6 = st.columns(3)

    with p1:
        in_cat = st.selectbox("📂 Category",
                              sorted(df['Category'].dropna().unique()))
    with p2:
        sub_opts_ml = sorted(
            df[df['Category'] == in_cat]['Sub-Category'].dropna().unique())
        in_sub = st.selectbox("🏷️ Sub-Category", sub_opts_ml)
    with p3:
        in_seg = st.selectbox("👥 Segment",
                              sorted(df['Segment'].dropna().unique()))
    with p4:
        in_ship = st.selectbox("🚚 Ship Mode",
                               sorted(df['Ship Mode'].dropna().unique()))
    with p5:
        in_qty = st.number_input("📦 Quantity", min_value=1,
                                 max_value=100, value=3)
    with p6:
        in_disc = st.slider("🏷️ Discount", 0.0, 1.0, 0.2, 0.05,
                            format="%.2f")

    p7, p8 = st.columns(2)
    with p7:
        in_year = st.selectbox("📅 Year", list(range(2020, 2031)), index=5)
    with p8:
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        in_month_name = st.selectbox("📆 Month", month_names)
        month_num = month_names.index(in_month_name) + 1

    in_region = None
    if 'Region' in df.columns and 'Region' in cat_cols:
        in_region = st.selectbox("🌎 Region",
                                 sorted(df['Region'].dropna().unique()))

    if st.button("🚀  Predict Sales Now", use_container_width=True):
        val_map = {
            'Category':     in_cat,
            'Sub-Category': in_sub,
            'Segment':      in_seg,
            'Ship Mode':    in_ship,
            'Region':       in_region,
        }
        input_vals = {}
        for col in cat_cols:
            raw = val_map.get(col, 'Unknown')
            try:
                input_vals[col + '_enc'] = encoders[col].transform([str(raw)])[0]
            except Exception:
                input_vals[col + '_enc'] = 0

        input_vals['Quantity'] = in_qty
        input_vals['Discount'] = in_disc
        input_vals['Year']     = in_year
        input_vals['Month']    = month_num

        X_new = pd.DataFrame([input_vals]).reindex(
            columns=feature_cols, fill_value=0)
        predicted_sales = model.predict(X_new)[0]

        cat_margin = df[df['Category'] == in_cat]['Profit Margin'].mean() / 100
        est_profit  = predicted_sales * cat_margin * (1 - in_disc)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction Results")
        r1, r2_col, r3 = st.columns(3)
        r1.metric("💰 Predicted Sales",  f"${predicted_sales:,.2f}")
        r2_col.metric("📈 Est. Profit",  f"${est_profit:,.2f}")
        r3.metric("🏷️ Discount Applied", f"{in_disc*100:.0f}%")

        ok = est_profit > 0
        status_text  = "✅  This looks like a Profitable Order!" if ok \
                       else "⚠️  Risk of Loss — consider adjusting discount or quantity."
        status_color = "#4ade80" if ok else "#f87171"
        st.markdown(
            f"<div style='text-align:center;font-size:1.3rem;font-weight:700;"
            f"color:{status_color};padding:20px;background:rgba(0,0,0,0.3);"
            f"border-radius:12px;margin:16px 0;'>{status_text}</div>",
            unsafe_allow_html=True)

        hist_avg = df[
            (df['Category'] == in_cat) &
            (df['Sub-Category'] == in_sub)
        ]['Sales'].mean()

        comp = pd.DataFrame({
            'Type':  ['Historical Avg', 'Predicted'],
            'Sales': [hist_avg, predicted_sales]
        })
        fig_c = px.bar(comp, x='Type', y='Sales', color='Type',
                       color_discrete_sequence=['#8b5cf6','#4ade80'],
                       template='plotly_dark', text_auto='.2f')
        fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.markdown("#### 📊 Predicted vs Historical Average")
        st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("#### 🗂️ Similar Historical Orders")
        similar = df[
            (df['Category'] == in_cat) &
            (df['Sub-Category'] == in_sub) &
            (df['Segment'] == in_seg)
        ][['Order Date','Product Name','Segment','Ship Mode',
           'Quantity','Discount','Sales','Profit']].tail(10)
        if not similar.empty:
            st.dataframe(similar.reset_index(drop=True),
                         use_container_width=True)
        else:
            st.info("No historical orders match these exact filters.")
