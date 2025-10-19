import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats
import time
import warnings
import requests
import re
from bs4 import BeautifulSoup
import streamlit as st

warnings.filterwarnings('ignore')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"
}

# Page configuration
st.set_page_config(
    page_title="Fund Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - FIXED METRIC BOXES TO BLACK BACKGROUND WITH WHITE TEXT
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {
        background-color: #1a1a1a !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 1px solid #333 !important;
    }
    .stMetric label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 24px !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #4CAF50 !important;
    }
    h1 {color: #1f77b4;}
    h2 {color: #2ca02c;}
    </style>
    """, unsafe_allow_html=True)


# ===== ALL ANALYSIS FUNCTIONS =====

@st.cache_data(ttl=3600)
def get_fund_data(ticker, period="5y"):
    """Fetch fund data with caching"""
    fund = yf.Ticker(ticker)
    return fund.history(period=period)


@st.cache_data(ttl=3600)
def get_fund_info(ticker):
    """Fetch fund information from Yahoo Finance"""
    try:
        fund = yf.Ticker(ticker)
        info = fund.info

        fund_details = {
            'name': info.get('longName', info.get('shortName', ticker)),
            'description': info.get('longBusinessSummary', 'No description available'),
            'fund_family': info.get('fundFamily', 'N/A'),
            'category': info.get('category', 'N/A'),
            'net_assets': info.get('totalAssets', 'N/A'),
            'inception_date': info.get('fundInceptionDate', 'N/A'),
            'current_price': info.get('regularMarketPrice', info.get('navPrice', 'N/A'))
        }
        return fund_details
    except Exception as e:
        st.warning(f"Could not fetch fund info: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def get_related_tickers(ticker):
    """Fetch related tickers from Yahoo Finance"""
    try:
        fund = yf.Ticker(ticker)

        # Try to get recommendations/similar tickers
        related = []

        # Get some info about similar funds (this is a simplified approach)
        # In reality, you might need to scrape the Yahoo Finance page for "People Also Watch"
        info = fund.info

        # For now, return some common related tickers based on fund type
        if 'SP' in ticker.upper() or '500' in str(info.get('longName', '')):
            related = [
                {'ticker': 'SPY', 'name': 'SPDR S&P 500 ETF Trust'},
                {'ticker': 'VOO', 'name': 'Vanguard S&P 500 ETF'},
                {'ticker': 'IVV', 'name': 'iShares Core S&P 500 ETF'}
            ]
        else:
            related = [
                {'ticker': 'VTI', 'name': 'Vanguard Total Stock Market ETF'},
                {'ticker': 'SPY', 'name': 'SPDR S&P 500 ETF Trust'},
                {'ticker': 'QQQ', 'name': 'Invesco QQQ Trust'}
            ]

        # Get current prices for related tickers
        for item in related:
            try:
                rel_ticker = yf.Ticker(item['ticker'])
                rel_info = rel_ticker.info
                item['price'] = rel_info.get('regularMarketPrice', 'N/A')
            except:
                item['price'] = 'N/A'

        return related
    except Exception as e:
        return []


@st.cache_data(ttl=3600)
def get_holdings(ticker, type_of_fund="Mutual"):
    """Fetch fund holdings from Zacks"""
    holdings = []
    try:
        if type_of_fund == "ETF":
            url = f"https://www.zacks.com/funds/etf/{ticker}/holding"
            with requests.Session() as req:
                req.headers.update(headers)
                r = req.get(url)
                ETF_stocks = re.findall(r'funds\\\/etf\\\/(.*?)\\', r.text)
                ETF_weight = re.findall(r'<\\\/span><\\\/span><\\\/a>",(.*?), "<a class=\\\"report_document newwin\\',
                                        r.text)
                for stock, weight in zip(ETF_stocks, ETF_weight):
                    match = re.findall(r'"([\d\.,]+)"', weight)
                    if len(match) >= 2:
                        clean_weight = re.sub(r'[^\d\.]', '', match[1])
                        weight_val = float(clean_weight)
                        holdings.append({"symbol": stock, "weight": weight_val})
        else:
            url = f"https://www.zacks.com/funds/mutual-fund/quote/{ticker}/holding"
            with requests.Session() as req:
                req.headers.update(headers)
                r = req.get(url)
                mutual_stock = re.findall(r'\\\/mutual-fund\\\/quote\\\/(.*?)\\', r.text)
                mutual_weight = re.findall(r'"sr-only\\\"><\\\/span><\\\/span><\\\/a>",(.*?)%", "', r.text)
                for stock, weight in zip(mutual_stock, mutual_weight):
                    match = re.findall(r'"([\d\.,]+)"?', weight)
                    if len(match) >= 5:
                        clean_weight = re.sub(r'[^\d\.]', '', match[-1])
                        weight_val = float(clean_weight)
                        holdings.append({"symbol": stock, "weight": weight_val})
    except Exception as e:
        st.warning(f"Error fetching holdings: {str(e)}")

    return holdings


@st.cache_data(ttl=3600)
def get_benchmark_data(benchmark="SPY", period="5y"):
    """Fetch benchmark data"""
    bench = yf.Ticker(benchmark)
    return bench.history(period=period)


@st.cache_data(ttl=3600)
def get_treasury_rate(period="5y"):
    """Fetch 10-Year Treasury rate (^TNX) as risk-free rate proxy"""
    try:
        tnx = yf.Ticker("^TNX")
        tnx_data = tnx.history(period=period)
        if not tnx_data.empty:
            # TNX is already in percentage, convert to decimal
            tnx_data['Rate'] = tnx_data['Close'] / 100
            return tnx_data
        return None
    except Exception as e:
        st.warning(f"Could not fetch Treasury rate: {str(e)}")
        return None


def calculate_returns(prices):
    """Calculate daily returns"""
    return prices.pct_change().dropna()


def calculate_sharpe_ratio(returns, risk_free_rate=0.04):
    """Calculate Sharpe Ratio (annualized)"""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_sortino_ratio(returns, risk_free_rate=0.04):
    """Calculate Sortino Ratio (annualized)"""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
    if downside_std == 0:
        return np.nan
    return np.sqrt(252) * excess_returns.mean() / downside_std


def calculate_max_drawdown(prices):
    """Calculate Maximum Drawdown"""
    cumulative = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_alpha_beta(fund_returns, benchmark_returns, risk_free_rate=0.04):
    """Calculate Alpha and Beta"""
    aligned_data = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
    if len(aligned_data) < 30:
        return np.nan, np.nan

    fund_ret = aligned_data.iloc[:, 0]
    bench_ret = aligned_data.iloc[:, 1]

    covariance = np.cov(fund_ret, bench_ret)[0][1]
    benchmark_variance = np.var(bench_ret)
    beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan

    fund_mean = fund_ret.mean() * 252
    bench_mean = bench_ret.mean() * 252
    alpha = fund_mean - (risk_free_rate + beta * (bench_mean - risk_free_rate))

    return alpha, beta


def calculate_var_cvar(returns, confidence_level=0.95):
    """Calculate Value at Risk and Conditional VaR"""
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar


def calculate_information_ratio(fund_returns, benchmark_returns):
    """Calculate Information Ratio"""
    aligned_data = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
    if len(aligned_data) < 30:
        return np.nan

    excess_returns = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
    tracking_error = excess_returns.std()

    if tracking_error == 0:
        return np.nan

    return np.sqrt(252) * excess_returns.mean() / tracking_error


def calculate_calmar_ratio(returns, prices):
    """Calculate Calmar Ratio"""
    annual_return = (1 + returns.mean()) ** 252 - 1
    max_dd = abs(calculate_max_drawdown(prices))
    return annual_return / max_dd if max_dd != 0 else np.nan


def calculate_omega_ratio(returns, threshold=0):
    """Calculate Omega Ratio"""
    returns_above = returns[returns > threshold] - threshold
    returns_below = threshold - returns[returns < threshold]

    if returns_below.sum() == 0:
        return np.nan

    return returns_above.sum() / returns_below.sum()


def stress_test_portfolio(returns, scenarios):
    """Stress test portfolio under various market conditions"""
    results = {}

    for name, shock in scenarios.items():
        stressed_returns = returns * (1 + shock)
        results[name] = {
            'return': stressed_returns.mean() * 252,
            'volatility': stressed_returns.std() * np.sqrt(252),
            'max_drawdown': calculate_max_drawdown(pd.Series((1 + stressed_returns).cumprod()))
        }

    return results


@st.cache_data(ttl=3600)
def get_holdings_returns(holdings_df, period="5y"):
    """Fetch returns for all holdings"""
    returns_data = {}

    # Limit to top 50 holdings for performance
    top_holdings = holdings_df.head(50)

    for idx, row in top_holdings.iterrows():
        ticker = row['symbol']
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                returns_data[ticker] = calculate_returns(hist['Close'])
        except:
            continue

    return returns_data


def calculate_efficient_frontier(returns_dict, num_portfolios=5000):
    """Calculate efficient frontier from holdings returns"""
    if len(returns_dict) < 2:
        return None, None, None

    # Create returns dataframe
    returns_df = pd.DataFrame(returns_dict).dropna()

    if returns_df.empty or len(returns_df.columns) < 2:
        return None, None, None

    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    num_assets = len(mean_returns)
    results = np.zeros((4, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.sum(weights * mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = portfolio_return / portfolio_std  # Sharpe ratio

    return results, mean_returns, cov_matrix


def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.04):
    """Find optimal portfolio weights to maximize Sharpe ratio"""
    num_assets = len(mean_returns)

    def neg_sharpe(weights):
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_std

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x


# ===== STREAMLIT APP =====

st.title("📊 Fund Analysis Tool")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    ticker = st.text_input("Fund Ticker", value="FXAIX", help="Enter the fund ticker symbol")
    fund_type = st.selectbox("Fund Type", ["Mutual", "ETF"], help="Select the type of fund")

    time_period = st.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=5,
        help="Select the historical period to analyze"
    )

    benchmark = st.text_input("Benchmark Ticker", value="SPY", help="Enter benchmark ticker for comparison")

    use_dynamic_rf = st.checkbox("Use Dynamic Risk-Free Rate (10Y Treasury)", value=True,
                                 help="Use ^TNX as dynamic risk-free rate instead of static 4%")

    analyze_button = st.button("🔍 Analyze Fund", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### About V2")
    st.info("Enhanced with dynamic risk-free rates, efficient frontier analysis, and detailed fund information.")

# Main analysis
if analyze_button:
    with st.spinner(f"Analyzing {ticker}..."):

        # Fetch all data
        fund_df = get_fund_data(ticker, time_period)
        bench_df = get_benchmark_data(benchmark, time_period)
        fund_info = get_fund_info(ticker)
        related_tickers = get_related_tickers(ticker)

        if fund_df is None or fund_df.empty:
            st.error(f"Unable to fetch data for {ticker}")
            st.stop()

        if bench_df is None or bench_df.empty:
            st.error(f"Unable to fetch benchmark data for {benchmark}")
            st.stop()

        # Get Treasury rate data
        tnx_data = get_treasury_rate(time_period) if use_dynamic_rf else None

        # Calculate average risk-free rate for the period
        if tnx_data is not None and not tnx_data.empty:
            avg_risk_free_rate = tnx_data['Rate'].mean()
            current_rf_rate = tnx_data['Rate'].iloc[-1]
        else:
            avg_risk_free_rate = 0.04
            current_rf_rate = 0.04

        # Calculate returns
        fund_returns = calculate_returns(fund_df['Close'])
        bench_returns = calculate_returns(bench_df['Close'])

        # Calculate all metrics with dynamic risk-free rate
        sharpe = calculate_sharpe_ratio(fund_returns, avg_risk_free_rate)
        sortino = calculate_sortino_ratio(fund_returns, avg_risk_free_rate)
        calmar = calculate_calmar_ratio(fund_returns, fund_df['Close'])
        alpha, beta = calculate_alpha_beta(fund_returns, bench_returns, avg_risk_free_rate)
        omega = calculate_omega_ratio(fund_returns)
        information = calculate_information_ratio(fund_returns, bench_returns)
        max_draw = calculate_max_drawdown(fund_df['Close'])
        var_95, cvar_95 = calculate_var_cvar(fund_returns, 0.95)

        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📈 Overview",
            "📊 Performance Metrics",
            "🎯 Holdings",
            "📉 Risk Analysis",
            "🔄 Rolling Metrics",
            "⚖️ Benchmark Comparison",
            "⚠️ Stress Testing",
            "💹 Risk-Free Rate Analysis",
            "🎯 Efficient Frontier"
        ])

        # ===== TAB 1: OVERVIEW (ENHANCED WITH FUND INFO) =====
        with tab1:
            st.header(f"{ticker} - Fund Overview")

            # Fund Information Section
            if fund_info:
                st.subheader("📋 Fund Details")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**{fund_info['name']}**")
                    st.markdown(f"*{fund_info['description'][:500]}...*" if len(
                        fund_info['description']) > 500 else f"*{fund_info['description']}*")

                with col2:
                    if fund_info['net_assets'] != 'N/A':
                        try:
                            net_assets_formatted = f"${fund_info['net_assets']:,.0f}"
                        except:
                            net_assets_formatted = str(fund_info['net_assets'])
                    else:
                        net_assets_formatted = 'N/A'

                    st.metric("Fund Family", fund_info['fund_family'])
                    st.metric("Category", fund_info['category'])
                    st.metric("Net Assets", net_assets_formatted)

                    if fund_info['inception_date'] != 'N/A':
                        try:
                            inception_str = datetime.fromtimestamp(fund_info['inception_date']).strftime('%Y-%m-%d')
                        except:
                            inception_str = str(fund_info['inception_date'])
                        st.metric("Inception Date", inception_str)

                st.markdown("---")

            # Related Tickers Section
            if related_tickers:
                st.subheader("🔗 Related Tickers")
                rel_cols = st.columns(len(related_tickers))
                for idx, rel in enumerate(related_tickers):
                    with rel_cols[idx]:
                        price_str = f"${rel['price']:.2f}" if isinstance(rel['price'], (int, float)) else str(
                            rel['price'])
                        st.metric(label=rel['name'][:30], value=f"{rel['ticker']}: {price_str}")

                st.markdown("---")

            # Performance Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${fund_df['Close'][-1]:.2f}")
            with col2:
                total_return = ((fund_df['Close'][-1] / fund_df['Close'][0]) - 1) * 100
                st.metric("Total Return", f"{total_return:.2f}%")
            with col3:
                ann_return = ((fund_df['Close'][-1] / fund_df['Close'][0]) ** (252 / len(fund_df)) - 1) * 100
                st.metric("Annualized Return", f"{ann_return:.2f}%")
            with col4:
                volatility = fund_returns.std() * np.sqrt(252) * 100
                st.metric("Volatility", f"{volatility:.2f}%")

            st.markdown("---")

            # Price charts
            col1, col2 = st.columns(2)

            with col1:
                close_price_fund = fund_df[['Close']].reset_index()
                close_price_fund.columns = ['Date', 'Close Price']
                fund_fig = px.line(
                    close_price_fund,
                    x='Date',
                    y='Close Price',
                    title=f'{ticker} - Close Price History',
                    labels={'Close Price': 'Price ($)', 'Date': 'Date'}
                )
                fund_fig.update_traces(line_color='#1f77b4')
                st.plotly_chart(fund_fig, use_container_width=True)

            with col2:
                close_price_bench = bench_df[['Close']].reset_index()
                close_price_bench.columns = ['Date', 'Close Price']
                bench_fig = px.line(
                    close_price_bench,
                    x='Date',
                    y='Close Price',
                    title=f'{benchmark} - Close Price History',
                    labels={'Close Price': 'Price ($)', 'Date': 'Date'}
                )
                bench_fig.update_traces(line_color='#ff7f0e')
                st.plotly_chart(bench_fig, use_container_width=True)

        # ===== TAB 2: PERFORMANCE METRICS =====
        with tab2:
            st.header("Performance Metrics")

            st.info(
                f"Risk-Free Rate Used: {avg_risk_free_rate * 100:.2f}% (Average) | Current: {current_rf_rate * 100:.2f}%")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            with col2:
                st.metric("Sortino Ratio", f"{sortino:.3f}")
            with col3:
                st.metric("Calmar Ratio", f"{calmar:.3f}")
            with col4:
                st.metric("Omega Ratio", f"{omega:.3f}")

            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Alpha", f"{alpha * 100:.2f}%")
            with col6:
                st.metric("Beta", f"{beta:.3f}")
            with col7:
                st.metric("Information Ratio", f"{information:.3f}")
            with col8:
                st.metric("Max Drawdown", f"{max_draw * 100:.2f}%")

            st.markdown("---")

            # Detailed metrics table
            st.subheader("Detailed Performance Summary")
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Total Return',
                    'Annualized Return',
                    'Volatility (Annual)',
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Calmar Ratio',
                    'Omega Ratio',
                    'Alpha',
                    'Beta',
                    'Information Ratio',
                    'Max Drawdown',
                    'VaR (95%)',
                    'CVaR (95%)'
                ],
                'Value': [
                    f"{total_return:.2f}%",
                    f"{ann_return:.2f}%",
                    f"{volatility:.2f}%",
                    f"{sharpe:.3f}",
                    f"{sortino:.3f}",
                    f"{calmar:.3f}",
                    f"{omega:.3f}",
                    f"{alpha * 100:.2f}%",
                    f"{beta:.3f}",
                    f"{information:.3f}",
                    f"{max_draw * 100:.2f}%",
                    f"{var_95 * 100:.2f}%",
                    f"{cvar_95 * 100:.2f}%"
                ]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # ===== TAB 3: HOLDINGS =====
        with tab3:
            st.header("Fund Holdings")

            with st.spinner("Fetching holdings data..."):
                fund_holdings = get_holdings(ticker, fund_type)

            if fund_holdings:
                holdings_df = pd.DataFrame(fund_holdings)
                holdings_df['weight'] = holdings_df['weight'].astype(float)
                holdings_df = holdings_df.sort_values(by='weight', ascending=False).reset_index(drop=True)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader(f"Top Holdings (Total: {len(holdings_df)})")
                    st.dataframe(
                        holdings_df.head(25).style.format({'weight': '{:.2f}%'}),
                        use_container_width=True,
                        height=600
                    )

                with col2:
                    # Top 10 pie chart
                    top_10 = holdings_df.head(10)
                    fig = px.pie(
                        top_10,
                        values='weight',
                        names='symbol',
                        title='Top 10 Holdings Allocation'
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

                    # Summary stats
                    st.metric("Total Holdings", len(holdings_df))
                    st.metric("Top 10 Concentration", f"{holdings_df.head(10)['weight'].sum():.2f}%")
                    st.metric("Largest Holding",
                              f"{holdings_df.iloc[0]['symbol']} ({holdings_df.iloc[0]['weight']:.2f}%)")

                    # Top holdings bar chart
                    fig = px.bar(
                        holdings_df.head(10),
                        x='weight',
                        y='symbol',
                        orientation='h',
                        title='Top 10 Holdings by Weight'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No holdings data available.")

        # ===== TAB 4: RISK ANALYSIS =====
        with tab4:
            st.header("Risk Analysis")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Drawdown", f"{max_draw * 100:.2f}%")
            with col2:
                st.metric("VaR (95%)", f"{var_95 * 100:.2f}%")
            with col3:
                st.metric("CVaR (95%)", f"{cvar_95 * 100:.2f}%")

            st.markdown("---")

            # Drawdown chart
            cumulative = (1 + fund_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='#d62728', width=2)
            ))
            fig.update_layout(
                title="Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Rolling max drawdown
            rolling_max_draw = fund_df['Close'].rolling(window=252).apply(calculate_max_drawdown).dropna()
            rolling_max_draw_plot = px.line(rolling_max_draw, title='Rolling Max Drawdown (1 Year Window)')
            rolling_max_draw_plot.update_traces(line_color='#d62728')
            st.plotly_chart(rolling_max_draw_plot, use_container_width=True)

            # Returns distribution
            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    fund_returns * 100,
                    nbins=50,
                    title='Distribution of Daily Returns',
                    labels={'value': 'Daily Return (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Q-Q plot for normality check
                from scipy.stats import probplot

                qq_data = probplot(fund_returns.dropna(), dist="norm")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[0][1],
                    mode='markers',
                    name='Returns'
                ))
                fig.add_trace(go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                    mode='lines',
                    name='Normal',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title='Q-Q Plot (Normality Check)',
                    xaxis_title='Theoretical Quantiles',
                    yaxis_title='Sample Quantiles'
                )
                st.plotly_chart(fig, use_container_width=True)

        # ===== TAB 5: ROLLING METRICS =====
        with tab5:
            st.header("Rolling Performance Metrics")

            # Rolling Sharpe
            rolling_sharpe = fund_returns.rolling(window=252).apply(
                lambda x: calculate_sharpe_ratio(x, avg_risk_free_rate)
            ).dropna()
            rolling_sharpe_plot = px.line(rolling_sharpe, title='Rolling Sharpe Ratio (1 Year Window)')
            rolling_sharpe_plot.update_traces(line_color='#2ca02c')
            st.plotly_chart(rolling_sharpe_plot, use_container_width=True)

            # Rolling Beta
            rolling_beta = (
                    fund_returns.rolling(window=252).cov(bench_returns) /
                    bench_returns.rolling(window=252).var()
            ).dropna()
            rolling_beta_plot = px.line(rolling_beta, title='Rolling Beta (1 Year Window)')
            rolling_beta_plot.update_traces(line_color='#ff7f0e')
            st.plotly_chart(rolling_beta_plot, use_container_width=True)

            # Rolling Alpha
            rolling_alpha = (
                                    fund_returns.rolling(window=252).mean() -
                                    bench_returns.rolling(window=252).mean()
                            ) * 252
            rolling_alpha_plot = px.line(rolling_alpha, title='Rolling Alpha (1 Year Window)')
            rolling_alpha_plot.update_traces(line_color='#9467bd')
            st.plotly_chart(rolling_alpha_plot, use_container_width=True)

            # Rolling volatility
            rolling_vol = fund_returns.rolling(window=252).std() * np.sqrt(252) * 100
            rolling_vol_plot = px.line(rolling_vol, title='Rolling Volatility (1 Year Window)')
            rolling_vol_plot.update_traces(line_color='#d62728')
            rolling_vol_plot.update_layout(yaxis_title='Volatility (%)')
            st.plotly_chart(rolling_vol_plot, use_container_width=True)

        # ===== TAB 6: BENCHMARK COMPARISON =====
        with tab6:
            st.header(f"Comparison with {benchmark}")

            # Key comparison metrics
            bench_sharpe = calculate_sharpe_ratio(bench_returns, avg_risk_free_rate)
            bench_sortino = calculate_sortino_ratio(bench_returns, avg_risk_free_rate)
            bench_max_draw = calculate_max_drawdown(bench_df['Close'])
            bench_vol = bench_returns.std() * np.sqrt(252) * 100

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Alpha", f"{alpha * 100:.2f}%")
            with col2:
                st.metric("Beta", f"{beta:.3f}")
            with col3:
                st.metric("Information Ratio", f"{information:.3f}")
            with col4:
                correlation = fund_returns.corr(bench_returns)
                st.metric("Correlation", f"{correlation:.3f}")
            with col5:
                tracking_error = (fund_returns - bench_returns).std() * np.sqrt(252) * 100
                st.metric("Tracking Error", f"{tracking_error:.2f}%")

            st.markdown("---")

            # Normalized performance
            fund_normalized = (fund_df['Close'] / fund_df['Close'].iloc[0]) * 100
            bench_normalized = (bench_df['Close'] / bench_df['Close'].iloc[0]) * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fund_normalized.index,
                y=fund_normalized,
                mode='lines',
                name=ticker,
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=bench_normalized.index,
                y=bench_normalized,
                mode='lines',
                name=benchmark,
                line=dict(color='#ff7f0e', width=2)
            ))
            fig.update_layout(
                title="Normalized Performance Comparison (Base 100)",
                xaxis_title="Date",
                yaxis_title="Indexed Value",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Side-by-side comparison
            st.subheader("Performance Metrics Comparison")
            bench_total_return = ((bench_df['Close'][-1] / bench_df['Close'][0]) - 1) * 100
            bench_ann_return = ((bench_df['Close'][-1] / bench_df['Close'][0]) ** (252 / len(bench_df)) - 1) * 100

            comparison_df = pd.DataFrame({
                'Metric': [
                    'Total Return',
                    'Annualized Return',
                    'Volatility',
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Max Drawdown'
                ],
                ticker: [
                    f"{total_return:.2f}%",
                    f"{ann_return:.2f}%",
                    f"{volatility:.2f}%",
                    f"{sharpe:.3f}",
                    f"{sortino:.3f}",
                    f"{max_draw * 100:.2f}%"
                ],
                benchmark: [
                    f"{bench_total_return:.2f}%",
                    f"{bench_ann_return:.2f}%",
                    f"{bench_vol:.2f}%",
                    f"{bench_sharpe:.3f}",
                    f"{bench_sortino:.3f}",
                    f"{bench_max_draw * 100:.2f}%"
                ]
            })
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # ===== TAB 7: STRESS TESTING =====
        with tab7:
            st.header("Stress Testing")
            st.markdown("Analyze how the fund would perform under historical market crash scenarios")

            scenarios = {
                '2008 Financial Crisis': -0.40,
                'COVID-19 Crash (2020)': -0.35,
                'Dot-com Bubble': -0.30,
                'Black Monday (1987)': -0.25,
                'Moderate Correction': -0.15,
                'Flash Crash': -0.20,
                'Minor Correction': -0.10
            }

            stress_results = stress_test_portfolio(fund_returns, scenarios)

            # Convert to DataFrame
            stress_df = pd.DataFrame(stress_results).T
            stress_df['return'] = stress_df['return'] * 100
            stress_df['volatility'] = stress_df['volatility'] * 100
            stress_df['max_drawdown'] = stress_df['max_drawdown'] * 100
            stress_df = stress_df.round(2)
            stress_df.columns = ['Expected Return (%)', 'Volatility (%)', 'Max Drawdown (%)']

            st.dataframe(stress_df, use_container_width=True)

            # Visualize stress test results
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    stress_df.reset_index(),
                    x='index',
                    y='Expected Return (%)',
                    title='Expected Returns Under Stress Scenarios',
                    labels={'index': 'Scenario'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(
                    stress_df.reset_index(),
                    x='index',
                    y='Max Drawdown (%)',
                    title='Max Drawdown Under Stress Scenarios',
                    labels={'index': 'Scenario'},
                    color='Max Drawdown (%)',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            # Worst case analysis
            st.markdown("---")
            st.subheader("Worst Case Analysis")
            worst_scenario = stress_df['Expected Return (%)'].idxmin()
            worst_return = stress_df.loc[worst_scenario, 'Expected Return (%)']
            worst_drawdown = stress_df.loc[worst_scenario, 'Max Drawdown (%)']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Worst Scenario", worst_scenario)
            with col2:
                st.metric("Expected Return", f"{worst_return:.2f}%")
            with col3:
                st.metric("Max Drawdown", f"{worst_drawdown:.2f}%")

        # ===== TAB 8: RISK-FREE RATE ANALYSIS (NEW) =====
        with tab8:
            st.header("💹 Risk-Free Rate Analysis (10-Year Treasury)")

            if tnx_data is not None and not tnx_data.empty:
                st.success(f"Current 10Y Treasury Rate: {current_rf_rate * 100:.2f}%")
                st.info(f"Average Rate Over Period: {avg_risk_free_rate * 100:.2f}%")

                # Plot Treasury rate over time
                tnx_plot_data = tnx_data[['Rate']].reset_index()
                tnx_plot_data.columns = ['Date', 'Rate']
                tnx_plot_data['Rate'] = tnx_plot_data['Rate'] * 100

                fig = px.line(
                    tnx_plot_data,
                    x='Date',
                    y='Rate',
                    title='10-Year Treasury Rate Over Time',
                    labels={'Rate': 'Rate (%)'}
                )
                fig.update_traces(line_color='#ff7f0e')
                fig.add_hline(y=avg_risk_free_rate * 100, line_dash="dash",
                              annotation_text=f"Average: {avg_risk_free_rate * 100:.2f}%",
                              line_color="red")
                st.plotly_chart(fig, use_container_width=True)

                # Statistics
                st.subheader("Treasury Rate Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Rate", f"{current_rf_rate * 100:.2f}%")
                with col2:
                    st.metric("Average Rate", f"{avg_risk_free_rate * 100:.2f}%")
                with col3:
                    st.metric("Min Rate", f"{tnx_data['Rate'].min() * 100:.2f}%")
                with col4:
                    st.metric("Max Rate", f"{tnx_data['Rate'].max() * 100:.2f}%")

                # Impact on Sharpe Ratio
                st.markdown("---")
                st.subheader("Impact of Risk-Free Rate on Sharpe Ratio")

                # Calculate Sharpe at different rates
                rate_scenarios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
                sharpe_scenarios = [calculate_sharpe_ratio(fund_returns, r) for r in rate_scenarios]

                sharpe_comparison = pd.DataFrame({
                    'Risk-Free Rate (%)': [r * 100 for r in rate_scenarios],
                    'Sharpe Ratio': sharpe_scenarios
                })

                fig = px.line(
                    sharpe_comparison,
                    x='Risk-Free Rate (%)',
                    y='Sharpe Ratio',
                    title='Sharpe Ratio Sensitivity to Risk-Free Rate',
                    markers=True
                )
                fig.add_vline(x=avg_risk_free_rate * 100, line_dash="dash",
                              annotation_text=f"Current Avg: {avg_risk_free_rate * 100:.2f}%",
                              line_color="red")
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Unable to fetch Treasury rate data. Using static 4% risk-free rate.")
                st.info("Using default risk-free rate of 4% for all calculations.")

        # ===== TAB 9: EFFICIENT FRONTIER (NEW) =====
        with tab9:
            st.header("🎯 Efficient Frontier Analysis")
            st.markdown("Comparing fund holdings allocation vs. optimal portfolio")

            if fund_holdings:
                with st.spinner("Analyzing holdings returns and calculating efficient frontier..."):
                    # Get returns for holdings
                    returns_dict = get_holdings_returns(holdings_df, time_period)

                    if len(returns_dict) >= 2:
                        # Calculate efficient frontier
                        results, mean_returns, cov_matrix = calculate_efficient_frontier(returns_dict)

                        if results is not None:
                            # Find optimal portfolio
                            optimal_weights = optimize_portfolio(mean_returns, cov_matrix, avg_risk_free_rate)

                            # Calculate fund's actual position
                            fund_holdings_dict = {}
                            for idx, row in holdings_df.iterrows():
                                if row['symbol'] in returns_dict:
                                    fund_holdings_dict[row['symbol']] = row['weight'] / 100

                            # Calculate fund metrics
                            if len(fund_holdings_dict) > 0:
                                fund_weights = np.array(
                                    [fund_holdings_dict.get(ticker, 0) for ticker in mean_returns.index])
                                fund_weights = fund_weights / fund_weights.sum() if fund_weights.sum() > 0 else fund_weights

                                fund_portfolio_return = np.sum(fund_weights * mean_returns)
                                fund_portfolio_std = np.sqrt(np.dot(fund_weights.T, np.dot(cov_matrix, fund_weights)))
                                fund_sharpe = (fund_portfolio_return - avg_risk_free_rate) / fund_portfolio_std
                            else:
                                fund_portfolio_return = np.nan
                                fund_portfolio_std = np.nan
                                fund_sharpe = np.nan

                            # Calculate optimal portfolio metrics
                            optimal_return = np.sum(optimal_weights * mean_returns)
                            optimal_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                            optimal_sharpe = (optimal_return - avg_risk_free_rate) / optimal_std

                            # Display metrics
                            st.subheader("Portfolio Comparison")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Current Fund Allocation**")
                                if not np.isnan(fund_portfolio_return):
                                    st.metric("Expected Return", f"{fund_portfolio_return * 100:.2f}%")
                                    st.metric("Volatility", f"{fund_portfolio_std * 100:.2f}%")
                                    st.metric("Sharpe Ratio", f"{fund_sharpe:.3f}")
                                else:
                                    st.warning("Unable to calculate fund metrics from holdings")

                            with col2:
                                st.markdown("**Optimal Portfolio (Max Sharpe)**")
                                st.metric("Expected Return", f"{optimal_return * 100:.2f}%")
                                st.metric("Volatility", f"{optimal_std * 100:.2f}%")
                                st.metric("Sharpe Ratio", f"{optimal_sharpe:.3f}")

                            # Plot efficient frontier
                            st.subheader("Efficient Frontier")

                            fig = go.Figure()

                            # Scatter plot of random portfolios
                            fig.add_trace(go.Scatter(
                                x=results[1, :],
                                y=results[0, :],
                                mode='markers',
                                marker=dict(
                                    size=5,
                                    color=results[2, :],
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Sharpe Ratio")
                                ),
                                name='Random Portfolios',
                                text=[f"Sharpe: {s:.3f}" for s in results[2, :]],
                                hovertemplate='Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>%{text}<extra></extra>'
                            ))

                            # Mark optimal portfolio
                            fig.add_trace(go.Scatter(
                                x=[optimal_std],
                                y=[optimal_return],
                                mode='markers',
                                marker=dict(size=20, color='red', symbol='star'),
                                name='Optimal Portfolio'
                            ))

                            # Mark current fund position (if calculable)
                            if not np.isnan(fund_portfolio_return):
                                fig.add_trace(go.Scatter(
                                    x=[fund_portfolio_std],
                                    y=[fund_portfolio_return],
                                    mode='markers',
                                    marker=dict(size=15, color='blue', symbol='diamond'),
                                    name='Current Fund'
                                ))

                            fig.update_layout(
                                title='Efficient Frontier',
                                xaxis_title='Volatility (Standard Deviation)',
                                yaxis_title='Expected Return',
                                hovermode='closest',
                                height=600
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Optimal weights table
                            st.subheader("Suggested Optimal Weights")
                            st.markdown("*To maximize Sharpe ratio based on historical returns*")

                            optimal_weights_df = pd.DataFrame({
                                'Ticker': mean_returns.index,
                                'Current Weight (%)': [fund_holdings_dict.get(ticker, 0) * 100 for ticker in
                                                       mean_returns.index],
                                'Optimal Weight (%)': optimal_weights * 100,
                                'Difference (%)': (optimal_weights * 100) - [fund_holdings_dict.get(ticker, 0) * 100 for
                                                                             ticker in mean_returns.index]
                            })

                            optimal_weights_df = optimal_weights_df.sort_values('Optimal Weight (%)', ascending=False)
                            optimal_weights_df = optimal_weights_df[
                                optimal_weights_df['Optimal Weight (%)'] > 0.1]  # Filter out tiny weights

                            st.dataframe(
                                optimal_weights_df.style.format({
                                    'Current Weight (%)': '{:.2f}',
                                    'Optimal Weight (%)': '{:.2f}',
                                    'Difference (%)': '{:+.2f}'
                                }),
                                use_container_width=True,
                                hide_index=True
                            )

                            st.info(
                                "⚠️ Note: This analysis is based on historical returns and may not predict future performance. Past performance does not guarantee future results.")

                        else:
                            st.warning(
                                "Unable to calculate efficient frontier. Need more holdings with available data.")
                    else:
                        st.warning(
                            f"Not enough holdings data available. Found {len(returns_dict)} holdings with return data. Need at least 2.")
            else:
                st.warning("No holdings data available for efficient frontier analysis.")

else:
    # Welcome screen
    st.info("👈 Configure your analysis in the sidebar and click 'Analyze Fund' to get started!")

    st.markdown("## 🚀 Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 Overview")
        st.write("• Fund description and details")
        st.write("• Price Charts and Performance Metrics")
        st.write("• Fund Holdings")
        st.write("• Benchmark Comparisons")

    with col2:
        st.markdown("### 💹 Risk Analysis")
        st.write("• Rolling Ratio Analysis")
        st.write("• Stress Testing")
        st.write("• Risk Free Rate Impact on Sharpe ratio")
        st.write("• Value at Risk and Max Drawdown")

    with col3:
        st.markdown("### 🎯 Efficient Frontier")
        st.write("• Holdings-based analysis")
        st.write("• Optimal portfolio weights")
        st.write("• Sharpe ratio maximization")
        st.write("• Portfolio Recommendations")

    st.markdown("---")

    st.markdown("## 📚 How to Use")
    st.markdown("""
    1. **Enter Fund Ticker**: Input the ticker symbol (e.g., FXAIX, VTSAX)
    2. **Select Fund Type**: Choose between Mutual Fund or ETF
    3. **Choose Time Period**: Select historical period (1mo to max)
    4. **Set Benchmark**: Enter benchmark ticker for comparison (e.g., SPY)
    5. **Enable Dynamic Risk-Free Rate**: Use real 10Y Treasury rates
    6. **Click Analyze**: Generate comprehensive analysis with all new features
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>Fund Analysis Tool V2</b></p>
    <p>Data provided by Yahoo Finance and Zacks</p>
    <p style='font-size: 12px;'>⚠️ This tool is for informational purposes only. Not financial advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)
