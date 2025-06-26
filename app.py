import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_card import card
import time
import calendar
import warnings
warnings.filterwarnings('ignore')

# Extended category keywords with emojis
CATEGORY_KEYWORDS = {
    '🍔 Food': ['restaurant', 'dinner', 'coffee', 'lunch', 'groceries', 'kfc', 'dominos', 'mcdonalds', 'burger', 'pizza'],
    '🚗 Transport': ['uber', 'taxi', 'bus', 'train', 'fuel', 'metro', 'ola', 'lyft', 'petrol', 'diesel'],
    '🛍️ Shopping': ['clothes', 'amazon', 'flipkart', 'store', 'mall', 'zara', 'h&m', 'myntra', 'ajio'],
    '💡 Utilities': ['electricity', 'water', 'internet', 'bill', 'mobile', 'phone', 'wifi', 'broadband'],
    '🏥 Health': ['medicine', 'doctor', 'pharmacy', 'hospital', 'clinic', 'med', 'pill', 'drug'],
    '🎬 Entertainment': ['movie', 'netflix', 'game', 'cinema', 'pvr', 'spotify', 'prime', 'disney', 'concert'],
    '🏠 Rent': ['rent', 'apartment', 'house', 'room', 'lease'],
    '✈️ Travel': ['flight', 'hotel', 'airbnb', 'vacation', 'trip', 'holiday'],
    '🎓 Education': ['book', 'course', 'tuition', 'school', 'college', 'university'],
    '💸 Investment': ['stock', 'mutual fund', 'crypto', 'bitcoin', 'etf', 'gold']
}

def categorize(description):
    if not isinstance(description, str):
        return 'Other'
    desc = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in desc for keyword in keywords):
            return category
    return 'Other'

def detect_outliers(df):
    if len(df) < 10:
        df['outlier'] = 1  # Not enough data for reliable outlier detection
        return df
    iso = IsolationForest(contamination=0.1, random_state=42)
    df['outlier'] = iso.fit_predict(df[['Amount']])
    return df

def forecast_expenses(df):
    if len(df) < 30:
        raise ValueError("Need at least 30 days of data for forecasting")
    df_prophet = df[['Date', 'Amount']].copy()
    df_prophet = df_prophet.dropna()
    df_prophet = df_prophet[df_prophet['Amount'] > 0]
    df_prophet = df_prophet.groupby('Date').sum().reset_index()
    df_prophet.columns = ['ds', 'y']
    model = Prophet(weekly_seasonality=True, daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=60)
    forecast = model.predict(future)
    return forecast, model

def animate_text(text, speed=0.05):
    placeholder = st.empty()
    for i in range(len(text) + 1):
        placeholder.markdown(f"## {text[:i]}|")
        time.sleep(speed)
    placeholder.markdown(f"## {text}")

# Initialize session state
if 'expenses' not in st.session_state:
    st.session_state.expenses = pd.DataFrame([
        {"Date": pd.to_datetime("2025-06-01"), "Description": "Lunch at KFC", "Amount": 450, "Category": "🍔 Food"},
        {"Date": pd.to_datetime("2025-06-02"), "Description": "Bus fare", "Amount": 40, "Category": "🚗 Transport"},
        {"Date": pd.to_datetime("2025-06-03"), "Description": "Electricity bill", "Amount": 1200, "Category": "💡 Utilities"},
        {"Date": pd.to_datetime("2025-06-04"), "Description": "Movie at PVR", "Amount": 600, "Category": "🎬 Entertainment"},
        {"Date": pd.to_datetime("2025-06-05"), "Description": "Online pharmacy", "Amount": 300, "Category": "🏥 Health"},
    ])

if 'budget' not in st.session_state:
    st.session_state.budget = 3000

# Page Config with fancy theme
st.set_page_config(
    page_title="Expensio - Advanced Analyzer", 
    layout="wide",
    page_icon="💸",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.st-emotion-cache-1y4p8pa {
    padding: 2rem 3rem;
}

.header-animation {
    animation: fadeIn 1.5s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.expense-card {
    transition: all 0.3s ease;
    border-radius: 15px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.expense-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.category-chip {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    background-color: #e0e0e0;
}

.positive-trend {
    color: #2ecc71;
    font-weight: bold;
}

.negative-trend {
    color: #e74c3c;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# Sidebar with enhanced UI
with st.sidebar:
    st.markdown("<div class='header-animation'>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=80)
    st.title("Expensio")
    st.caption("Your Smart Expense Companion")
    st.markdown("</div>", unsafe_allow_html=True)
    
    add_vertical_space(2)
    
    with st.expander("📤 Upload Expenses", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV (Date, Description, Amount)", type=['csv'], 
                                       help="Upload your expense data in CSV format with columns: Date, Description, Amount")
        if uploaded_file:
            with st.spinner("Processing your data..."):
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    df_upload['Date'] = pd.to_datetime(df_upload['Date'], errors='coerce')
                    df_upload = df_upload.dropna(subset=['Date', 'Amount'])
                    df_upload['Category'] = df_upload['Description'].apply(lambda x: categorize(str(x)))
                    st.session_state.expenses = pd.concat([st.session_state.expenses, df_upload], ignore_index=True)
                    st.success("✅ File uploaded and merged successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    with st.expander("📝 Add Expense", expanded=True):
        with st.form("manual_entry"):
            date = st.date_input("Date", value=datetime.date.today())
            description = st.text_input("Description", placeholder="e.g. Dinner at Italian Restaurant")
            amount = st.number_input("Amount", min_value=0.0, step=10.0, value=0.0)
            category = st.selectbox("Category", list(CATEGORY_KEYWORDS.keys()), 
                                   index=0, help="Select or let the system auto-categorize")
            submitted = st.form_submit_button("➕ Add Expense", use_container_width=True)
            
            if submitted and amount > 0:
                new_row = {
                    "Date": pd.to_datetime(date), 
                    "Description": description, 
                    "Amount": amount, 
                    "Category": category if description.strip() == "" else categorize(description)
                }
                st.session_state.expenses = pd.concat([st.session_state.expenses, pd.DataFrame([new_row])], ignore_index=True)
                st.success("✅ Expense added successfully!")
                time.sleep(0.5)
                st.rerun()

    with st.expander("💰 Set Budget", expanded=True):
        st.session_state.budget = st.number_input("Monthly Budget (₹)", min_value=100, 
                                                 value=st.session_state.budget, step=500)
        
    with st.expander("⚙️ Settings", expanded=False):
        st.checkbox("Show raw data", value=False, key="show_raw")
        st.checkbox("Enable animations", value=True, key="enable_animations")

# Main Content Area
st.markdown("<div class='header-animation'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=100)
with col2:
    st.title("Expensio Dashboard")
    st.caption("Track, Analyze, and Optimize Your Spending")
st.markdown("</div>", unsafe_allow_html=True)

# Add a divider with animation
st.divider()

df = st.session_state.expenses.copy()

if not df.empty:
    # Current month and year
    current_month = datetime.datetime.now().month
    current_year = datetime.datetime.now().year
    month_name = calendar.month_name[current_month]
    
    # Metrics Row with Cards
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        # Total Spent
        total_spent = df[df['Date'].dt.month == current_month]['Amount'].sum()
        with col1:
            with stylable_container(
                key="metric1",
                css_styles="""
                    {
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    }
                    """,
            ):
                st.metric(label="Total Spent", value=f"₹{total_spent:,.2f}", 
                         delta=f"₹{(total_spent - st.session_state.budget):,.2f} vs Budget")
        
        # Budget Remaining
        remaining = max(0, st.session_state.budget - total_spent)
        with col2:
            with stylable_container(
                key="metric2",
                css_styles="""
                    {
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    }
                    """,
            ):
                st.metric(label="Budget Remaining", value=f"₹{remaining:,.2f}", 
                         delta=f"{int((remaining/st.session_state.budget)*100)}% of budget left")
        
        # Daily Average
        days_passed = datetime.datetime.now().day
        daily_avg = total_spent / days_passed if days_passed > 0 else 0
        with col3:
            with stylable_container(
                key="metric3",
                css_styles="""
                    {
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    }
                    """,
            ):
                st.metric(label="Daily Average", value=f"₹{daily_avg:,.2f}", 
                         delta="₹{0:,.2f} yesterday" if daily_avg > 0 else None)
        
        # Top Category
        top_category = df[df['Date'].dt.month == current_month].groupby('Category')['Amount'].sum().idxmax() if not df.empty else "N/A"
        with col4:
            with stylable_container(
                key="metric4",
                css_styles="""
                    {
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    }
                    """,
            ):
                st.metric(label="Top Spending Category", value=top_category)
    
    style_metric_cards()
    
    # Budget Alert with visual indicator
    budget_percentage = (total_spent / st.session_state.budget) * 100 if st.session_state.budget > 0 else 0
    with st.container():
        st.subheader("💰 Budget Status")
        budget_col1, budget_col2 = st.columns([1, 4])
        
        with budget_col1:
            if budget_percentage < 50:
                st.success(f"🟢 {budget_percentage:.1f}% of budget used")
            elif 50 <= budget_percentage < 80:
                st.warning(f"🟡 {budget_percentage:.1f}% of budget used")
            else:
                st.error(f"🔴 {budget_percentage:.1f}% of budget used")
        
        with budget_col2:
            st.progress(min(budget_percentage / 100, 1.0))
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Trends", "⚠️ Alerts", "🔮 Forecast", "🗃️ Data"])
    
    with tab1:
        # Monthly Spending by Category
        st.subheader(f"📊 {month_name} Spending by Category")
        monthly_df = df[df['Date'].dt.month == current_month]
        if not monthly_df.empty:
            fig = px.pie(monthly_df, names='Category', values='Amount', 
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for current month")
        
        # Weekly Spending Heatmap
        st.subheader("📅 Weekly Spending Pattern")
        df['Day'] = df['Date'].dt.day_name()
        df['Week'] = df['Date'].dt.isocalendar().week
        pivot = df.pivot_table(index='Day', columns='Week', values='Amount', 
                              aggfunc='sum', fill_value=0)
        
        # Reorder days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex(days_order)
        
        fig = px.imshow(pivot, color_continuous_scale='Blues', 
                       labels=dict(x="Week", y="Day", color="Amount"))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Monthly Trend Analysis
        st.subheader("📈 Monthly Spending Trend")
        df['Month-Year'] = df['Date'].dt.to_period('M').astype(str)
        monthly_trend = df.groupby('Month-Year')['Amount'].sum().reset_index()
        
        if len(monthly_trend) > 1:
            fig = px.line(monthly_trend, x='Month-Year', y='Amount', 
                         markers=True, text='Amount',
                         title="Monthly Spending Over Time")
            fig.update_traces(texttemplate='₹%{y:,.0f}', textposition='top center')
            fig.update_layout(yaxis_title="Amount (₹)", xaxis_title="Month")
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate trend
            last_month = monthly_trend['Amount'].iloc[-1]
            prev_month = monthly_trend['Amount'].iloc[-2]
            trend = ((last_month - prev_month) / prev_month) * 100
            
            if trend > 0:
                st.warning(f"🚨 Spending increased by {trend:.1f}% compared to previous month")
            else:
                st.success(f"✅ Spending decreased by {abs(trend):.1f}% compared to previous month")
        else:
            st.info("Need at least 2 months of data for trend analysis")
        
        # Category Trends
        st.subheader("🔄 Category Trends Over Time")
        category_trend = df.groupby(['Month-Year', 'Category'])['Amount'].sum().reset_index()
        
        if len(category_trend) > 0:
            fig = px.bar(category_trend, x='Month-Year', y='Amount', color='Category',
                        barmode='stack', text='Amount',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(texttemplate='₹%{y:,.0f}', textposition='inside')
            fig.update_layout(yaxis_title="Amount (₹)", xaxis_title="Month")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Outlier Detection
        st.subheader("⚠️ Unusual Spending Patterns")
        with st.spinner("Analyzing your spending patterns..."):
            df = detect_outliers(df)
            outliers = df[df['outlier'] == -1]
            
            if not outliers.empty:
                st.warning(f"🔍 Found {len(outliers)} unusual expenses that stand out from your typical spending:")
                
                # Display outliers as cards
                for _, row in outliers.sort_values('Amount', ascending=False).head(5).iterrows():
                    with stylable_container(
                        key=f"outlier_{row['Date']}",
                        css_styles="""
                            {
                                background: linear-gradient(135deg, #fff5f5 0%, #ffebee 100%);
                                border-radius: 10px;
                                padding: 15px;
                                margin-bottom: 10px;
                                border-left: 5px solid #ff5252;
                            }
                            """,
                    ):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{row['Description']}**")
                            st.caption(f"📅 {row['Date'].strftime('%b %d, %Y')}")
                            st.markdown(f"<span class='category-chip'>{row['Category']}</span>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"<h3 style='color: #e74c3c; text-align: right;'>₹{row['Amount']:,.2f}</h3>", 
                                       unsafe_allow_html=True)
            else:
                st.success("✅ No unusual spending patterns detected!")
        
        # Budget Warning
        st.subheader("💰 Budget Warnings")
        if budget_percentage >= 80:
            st.error("🚨 You've used 80% or more of your monthly budget!")
        elif budget_percentage >= 50:
            st.warning("⚠️ You've used 50% or more of your monthly budget")
        else:
            st.success("🟢 Your spending is within safe budget limits")
        
        # Large Expense Warning
        large_expense_threshold = st.session_state.budget * 0.2  # 20% of budget
        large_expenses = monthly_df[monthly_df['Amount'] >= large_expense_threshold]
        
        if not large_expenses.empty:
            st.warning(f"⚠️ Found {len(large_expenses)} large expenses (≥20% of your budget):")
            for _, row in large_expenses.iterrows():
                st.markdown(f"- **{row['Description']}**: ₹{row['Amount']:,.2f} ({row['Category']})")
    
    with tab4:
        # Expense Forecast
        st.subheader("🔮 Future Expense Forecast")
        
        if len(df) >= 30:
            with st.spinner("Generating forecast... This may take a moment"):
                try:
                    forecast, model = forecast_expenses(df)
                    
                    # Plot forecast
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=model.history['ds'],
                        y=model.history['y'],
                        name='Actual',
                        line=dict(color='#3498db')
                    ))
                    
                    # Add forecast
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat'],
                        name='Forecast',
                        line=dict(color='#2ecc71', dash='dot')
                    ))
                    
                    # Add uncertainty interval
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(46, 204, 113, 0.2)',
                        line_color='rgba(255,255,255,0)',
                        name='Uncertainty'
                    ))
                    
                    fig.update_layout(
                        title="60-Day Expense Forecast",
                        yaxis_title="Amount (₹)",
                        xaxis_title="Date",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Monthly forecast summary
                    forecast['Month'] = forecast['ds'].dt.to_period('M')
                    monthly_forecast = forecast.groupby('Month')['yhat'].sum().reset_index()
                    
                    if not monthly_forecast.empty:
                        st.subheader("📅 Monthly Forecast Summary")
                        cols = st.columns(len(monthly_forecast))
                        
                        for idx, (_, row) in enumerate(monthly_forecast.iterrows()):
                            with cols[idx]:
                                with stylable_container(
                                    key=f"forecast_{row['Month']}",
                                    css_styles="""
                                        {
                                            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                                            border-radius: 10px;
                                            padding: 15px;
                                            text-align: center;
                                        }
                                        """,
                                ):
                                    st.metric(
                                        label=str(row['Month']),
                                        value=f"₹{row['yhat']:,.0f}"
                                    )
                    
                    # Savings recommendation
                    avg_monthly = monthly_forecast['yhat'].mean()
                    if avg_monthly > st.session_state.budget:
                        st.error(f"⚠️ Based on trends, your average monthly spending (₹{avg_monthly:,.0f}) exceeds your budget (₹{st.session_state.budget:,.0f}). Consider adjusting your budget or reducing expenses.")
                    else:
                        st.success(f"✅ Based on trends, your average monthly spending (₹{avg_monthly:,.0f}) is within your budget (₹{st.session_state.budget:,.0f}). Good job!")
                
                except Exception as e:
                    st.error(f"Forecasting error: {str(e)}")
        else:
            st.info("ℹ️ Need at least 30 days of data for accurate forecasting")
    
    with tab5:
        # Interactive Data Explorer
        st.subheader("📝 Expense Records")
        filtered_df = dataframe_explorer(df.drop(columns=['outlier', 'Day', 'Week'], errors='ignore'))
        st.dataframe(filtered_df, use_container_width=True)
        
        # Data export options
        st.subheader("📤 Export Data")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Download CSV",
                csv,
                "expensio_data.csv",
                "text/csv",
                key='download-csv'
            )
        
        with export_col2:
            if st.button("📊 Generate Report PDF", key='generate-pdf'):
                with st.spinner("Generating PDF report..."):
                    time.sleep(2)  # Simulate PDF generation
                    st.success("✅ PDF report generated! (This is a simulation)")
        
        # Data insights
        st.subheader("🔍 Quick Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Expenses", f"₹{df['Amount'].sum():,.2f}")
        
        with col2:
            st.metric("Average Expense", f"₹{df['Amount'].mean():,.2f}")
        
        with col3:
            st.metric("Most Frequent Category", df['Category'].mode()[0] if not df.empty else "N/A")
    
    # Spending Habit Analysis
    with st.expander("🧠 AI-Powered Spending Insights", expanded=False):
        st.subheader("🤖 Smart Analysis of Your Spending Habits")
        
        if len(df) > 10:
            # Peak spending days
            df['DayOfWeek'] = df['Date'].dt.day_name()
            peak_day = df.groupby('DayOfWeek')['Amount'].sum().idxmax()
            
            # Most expensive category
            expensive_cat = df.groupby('Category')['Amount'].sum().idxmax()
            
            # Time between expenses
            df = df.sort_values('Date')
            df['DaysBetween'] = df['Date'].diff().dt.days
            avg_days_between = df['DaysBetween'].mean()
            
            # Create insights
            st.markdown(f"""
            - 🗓️ **Your peak spending day is {peak_day}** - consider planning activities on other days to balance expenses
            - 💰 **{expensive_cat} is your most expensive category** - look for ways to optimize spending in this area
            - ⏱️ **You make expenses every {avg_days_between:.1f} days on average** - tracking helps identify spending patterns
            - 📉 **Your largest single expense was ₹{df['Amount'].max():,.2f}** - was this planned or impulsive?
            """)
            
            # Recommendation engine
            if expensive_cat in ['🛍️ Shopping', '🎬 Entertainment']:
                st.info("💡 Tip: Consider setting a separate budget for discretionary spending categories like Shopping and Entertainment")
            
            if budget_percentage > 70 and current_month != 12:
                st.warning("⚠️ Warning: You're spending at a rate that may exceed your monthly budget. Consider reviewing non-essential expenses.")
            
            if '🚗 Transport' in expensive_cat:
                st.info("💡 Tip: For frequent transport expenses, consider monthly passes or carpooling to reduce costs")
        else:
            st.info("Need more data (at least 10 entries) for meaningful insights")
    
    # Gamification element
    if st.session_state.enable_animations:
        with st.expander("🎮 Expense Challenge", expanded=False):
            st.subheader("🏆 Monthly Savings Challenge")
            
            # Calculate savings score (0-100)
            savings_score = min(100, max(0, 100 - (total_spent / st.session_state.budget) * 100)) if st.session_state.budget > 0 else 0
            
            st.markdown(f"""
            **Your Savings Score:** {savings_score:.0f}/100
            """)
            
            # Progress with emoji rewards
            if savings_score >= 80:
                st.success("🎉 Excellent! You're saving more than 20% of your budget")
                st.balloons()
            elif savings_score >= 50:
                st.warning("👍 Good effort! Try to save a bit more next month")
            else:
                st.error("💪 Challenge: Can you reduce expenses by 10% next month?")
            
            # Badges system
            st.subheader("🛡️ Badges Earned")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if len(df) >= 30:
                    card(
                        title="30-Day Tracker",
                        text="Logged expenses for 30 days",
                        image="https://cdn-icons-png.flaticon.com/512/3132/3132693.png",
                        styles={
                            "card": {
                                "width": "100%",
                                "height": "100%"
                            }
                        }
                    )
            
            with col2:
                if len(outliers) == 0 and len(df) > 10:
                    card(
                        title="Consistent Spender",
                        text="No unusual expenses detected",
                        image="https://cdn-icons-png.flaticon.com/512/3132/3132732.png",
                        styles={
                            "card": {
                                "width": "100%",
                                "height": "100%"
                            }
                        }
                    )
            
            with col3:
                if savings_score >= 80:
                    card(
                        title="Super Saver",
                        text="Saved 20%+ of your budget",
                        image="https://cdn-icons-png.flaticon.com/512/3132/3132779.png",
                        styles={
                            "card": {
                                "width": "100%",
                                "height": "100%"
                            }
                        }
                    )

else:
    # Empty state with call to action
    st.markdown("""
    <div style="text-align: center; padding: 5rem 0;">
        <h2>📊 No Expenses Tracked Yet</h2>
        <p style="font-size: 1.2rem;">Get started by uploading your expense data or adding expenses manually</p>
        <img src="https://cdn-icons-png.flaticon.com/512/3976/3976626.png" width="200" style="margin: 2rem 0;">
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("➕ Add Your First Expense", use_container_width=True, type="primary"):
        switch_page("Add Expense")  # This would navigate to the add expense section