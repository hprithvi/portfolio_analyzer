#I'll create an interactive Streamlit app with graphical inputs and outputs.

#```python
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import re
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

sys.path.append(str(Path(__file__).parent))
from database.multi_asset_db_manager import MultiAssetDBManager

# Set page config
st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_manager(_version=2):
    return MultiAssetDBManager(db_url=os.getenv('DATABASE_URL'), db_type='postgresql')


class MutualFundPortfolio:
    """
    A comprehensive mutual fund portfolio analyzer with Monte Carlo simulation
    """

    def __init__(self):
        self.portfolio = {}
        self.nav_data = {}
        self.returns_data = None
        self.correlation_matrix = None

    def add_fund(self, fund_name, allocation_percentage):
        #"""Add a mutual fund to the portfolio with its allocation"""
        if allocation_percentage < 0 or allocation_percentage > 100:
            raise ValueError("Allocation percentage must be between 0 and 100")

        self.portfolio[fund_name] = allocation_percentage / 100
        return True

    def validate_portfolio(self):
        #"""Check if total allocation equals 100%"""
        total = sum(self.portfolio.values()) * 100
        if not np.isclose(total, 100, atol=0.01):
            return False, total
        return True, total

    def fetch_price_from_db(self, isin, display_name, db_manager, start_date=None):
        """Fetch close prices from price_history table, store in self.nav_data."""
        df = db_manager.get_price_history(isin, start_date=start_date)
        if df.empty:
            return None
        df = df.set_index('date')
        price_series = df['close_price'].sort_index()
        price_series.name = display_name
        self.nav_data[display_name] = price_series
        return price_series

    def fetch_nav_data(self, fund_symbol, start_date=None, end_date=None):
        #"""Fetch historical NAV data for a mutual fund"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            import yfinance as yf
            data = yf.download(fund_symbol, start=start_date, end=end_date, progress=False)
            if data.empty:
                raise ValueError(f"No data found for {fund_symbol}")

            self.nav_data[fund_symbol] = data['Adj Close']
            return data['Adj Close']

        except Exception as e:
            st.error(f"Error fetching data for {fund_symbol}: {str(e)}")
            return None

    def fetch_indian_mf_data(self, scheme_code, start_date=None):
        #"""Fetch NAV data for Indian Mutual Funds using mfapi.in"""
        try:
            url = f"https://api.mfapi.in/mf/{scheme_code}"
            response = requests.get(url)
            data = response.json()

            if 'data' in data:
                df = pd.DataFrame(data['data'])
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
                df['nav'] = pd.to_numeric(df['nav'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)

                if start_date:
                    df = df[df.index >= start_date]

                self.nav_data[scheme_code] = df['nav']
                return df['nav']
            else:
                raise ValueError(f"Invalid scheme code: {scheme_code}")

        except Exception as e:
            st.error(f"Error fetching Indian MF data: {str(e)}")
            return None

    def calculate_returns(self):
        #"""Calculate daily returns for all funds in portfolio"""
        if not self.nav_data:
            raise ValueError("No NAV data available. Fetch data first.")

        returns_dict = {}
        for fund, nav_series in self.nav_data.items():
            returns_dict[fund] = nav_series.pct_change().dropna()

        self.returns_data = pd.DataFrame(returns_dict)
        self.returns_data = self.returns_data.dropna()

        return self.returns_data

    def calculate_portfolio_risk(self):
        #"""Calculate portfolio risk using weights and correlations"""
        if self.returns_data is None:
            self.calculate_returns()

        # Get correlation matrix
        self.correlation_matrix = self.returns_data.corr()

        # Get covariance matrix
        cov_matrix = self.returns_data.cov()

        # Portfolio weights
        weights = np.array([self.portfolio[fund] for fund in self.returns_data.columns])

        # Portfolio variance
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Annualize
        annualized_std = portfolio_std * np.sqrt(252)

        return {
            'daily_volatility': portfolio_std,
            'annual_volatility': annualized_std,
            'correlation_matrix': self.correlation_matrix,
            'covariance_matrix': cov_matrix
        }

    def monte_carlo_simulation(self, months=3, num_simulations=10000):
        #"""Run Monte Carlo simulation for portfolio returns"""
        if self.returns_data is None:
            self.calculate_returns()

        # Calculate portfolio daily returns
        weights = np.array([self.portfolio[fund] for fund in self.returns_data.columns])
        portfolio_returns = (self.returns_data * weights).sum(axis=1)

        # Calculate statistics
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()

        # Number of trading days
        trading_days = int(months * 21)

        # Run simulations
        simulations = np.zeros((num_simulations, trading_days))
        final_returns = np.zeros(num_simulations)

        for i in range(num_simulations):
            daily_returns = np.random.normal(mean_return, std_return, trading_days)
            cumulative_returns = np.cumprod(1 + daily_returns) - 1
            simulations[i] = cumulative_returns
            final_returns[i] = cumulative_returns[-1]

        # Calculate percentiles
        median_return = np.percentile(final_returns, 50)
        top_20_return = np.percentile(final_returns, 80)
        bottom_10_return = np.percentile(final_returns, 10)

        return {
            'months': months,
            'simulations': simulations,
            'final_returns': final_returns,
            'median': median_return,
            'top_20': top_20_return,
            'bottom_10': bottom_10_return,
            'mean': final_returns.mean(),
            'std': final_returns.std(),
            'percentile_95': np.percentile(final_returns, 95),
            'percentile_5': np.percentile(final_returns, 5),
            'mean_daily': mean_return,
            'std_daily': std_return
        }


    def plot_portfolio_allocation(portfolio_dict):
    #"""Create pie chart for portfolio allocation"""
        fig = go.Figure(data=[go.Pie(
        labels=list(portfolio_dict.keys()),
        values=[v*100 for v in portfolio_dict.values()],
        hole=0.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])

        fig.update_layout(
        title="Portfolio Allocation",
        height=400,
        showlegend=True
    )

        return fig


    def plot_correlation_heatmap(correlation_matrix):
    #"""Create correlation heatmap with alphabet labels and legend"""
        import string, textwrap
        fund_names = list(correlation_matrix.columns)
        labels = list(string.ascii_uppercase[:len(fund_names)])

        # Build legend with wrapped fund names
        legend_lines = []
        for lbl, name in zip(labels, fund_names):
            wrapped = "<br>    ".join(textwrap.wrap(name, width=50))
            legend_lines.append(f"<b>{lbl}</b>: {wrapped}")
        legend_text = "<br>".join(legend_lines)

        # Dynamic bottom margin based on number of funds
        bottom_margin = 60 + len(fund_names) * 22

        fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

        fig.update_layout(
        title="Fund Correlation Matrix",
        height=400 + bottom_margin,
        width=500,
        xaxis=dict(tickangle=0),
        margin=dict(b=bottom_margin, l=40, r=40),
        annotations=[
            dict(
                text=legend_text,
                xref="paper", yref="paper",
                x=0, y=-0.02,
                showarrow=False,
                align="left",
                font=dict(size=11, family="monospace"),
                xanchor="left", yanchor="top"
            )
        ]
    )

        return fig


    def plot_nav_history(nav_data):
    #"""Plot NAV history for all funds"""
        fig = go.Figure()

        for fund, nav in nav_data.items():
        # Normalize to 100 for comparison
            normalized = (nav / nav.iloc[0]) * 100
            fig.add_trace(go.Scatter(
            x=nav.index,
            y=normalized,
            name=fund,
            mode='lines'
        ))

        fig.update_layout(
        title="Normalized Price Performance (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        height=500,
        hovermode='x unified'
    )

        return fig


    def plot_monte_carlo_distribution(final_returns, median, top_20, bottom_10):
    #"""Plot distribution of Monte Carlo simulation results"""
        fig = go.Figure()

    # Histogram
        fig.add_trace(go.Histogram(
        x=final_returns * 100,
        nbinsx=50,
        name='Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))

    # Add vertical lines for percentiles
        fig.add_vline(x=median*100, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median*100:.2f}%")
        fig.add_vline(x=top_20*100, line_dash="dash", line_color="blue",
                  annotation_text=f"80th %ile: {top_20*100:.2f}%")
        fig.add_vline(x=bottom_10*100, line_dash="dash", line_color="red",
                  annotation_text=f"10th %ile: {bottom_10*100:.2f}%")

        fig.update_layout(
        title="Distribution of Simulated Returns",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        height=500,
        showlegend=True
    )

        return fig


    def plot_simulation_paths(simulations, num_paths=100):
    #"""Plot sample simulation paths"""
        fig = go.Figure()

    # Plot random sample of paths
        indices = np.random.choice(len(simulations), min(num_paths, len(simulations)), replace=False)

        for idx in indices:
            fig.add_trace(go.Scatter(
            y=simulations[idx] * 100,
            mode='lines',
            line=dict(width=0.5),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add mean path
        mean_path = simulations.mean(axis=0) * 100
        fig.add_trace(go.Scatter(
        y=mean_path,
        mode='lines',
        line=dict(color='red', width=3),
        name='Mean Path'
    ))

        fig.update_layout(
        title=f"Sample Simulation Paths (showing {min(num_paths, len(simulations))} of {len(simulations)})",
        xaxis_title="Trading Days",
        yaxis_title="Cumulative Return (%)",
        height=500
    )

        return fig


    def plot_risk_metrics(annual_volatility):
    #"""Create gauge chart for annual volatility"""
        fig = go.Figure()

        fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=annual_volatility * 100,
        title={'text': "Annual Volatility (%)"},
        gauge={
            'axis': {'range': [0, 50]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 15], 'color': "lightgreen"},
                {'range': [15, 25], 'color': "yellow"},
                {'range': [25, 50], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))

        fig.update_layout(height=350)

        return fig


    def plot_percentile_comparison(results_3m, results_6m):
    #"""Compare percentiles across different time horizons"""
        categories = ['Bottom 10%', 'Median', 'Top 20%', 'Mean']

        values_3m = [
        results_3m['bottom_10'] * 100,
        results_3m['median'] * 100,
        results_3m['top_20'] * 100,
        results_3m['mean'] * 100
    ]

        values_6m = [
        results_6m['bottom_10'] * 100,
        results_6m['median'] * 100,
        results_6m['top_20'] * 100,
        results_6m['mean'] * 100
    ]

        fig = go.Figure(data=[
        go.Bar(name='3 Months', x=categories, y=values_3m, marker_color='lightblue'),
        go.Bar(name='6 Months', x=categories, y=values_6m, marker_color='darkblue')
    ])

        fig.update_layout(
        title="Expected Returns Comparison: 3 Months vs 6 Months",
        xaxis_title="Percentile",
        yaxis_title="Return (%)",
        barmode='group',
        height=500
    )

        return fig


def generate_pdf_report(portfolio, risk_metrics, results_3m, results_6m, sharpe, max_drawdown):
    """Generate a PDF report of the portfolio analysis."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Page 1: Portfolio Overview ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'Portfolio Analysis Report', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 8, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(10)

    # Portfolio allocation table
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Portfolio Allocation', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(120, 8, 'Asset', border=1, fill=True)
    pdf.cell(50, 8, 'Allocation (%)', border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_font('Helvetica', '', 10)
    for name, weight in portfolio.portfolio.items():
        # Truncate long names
        display = name[:55] + '...' if len(name) > 55 else name
        pdf.cell(120, 8, display, border=1)
        pdf.cell(50, 8, f'{weight*100:.1f}%', border=1, align='C')
        pdf.ln()

    pdf.ln(5)

    # Allocation pie chart
    try:
        fig_alloc = MutualFundPortfolio.plot_portfolio_allocation(portfolio.portfolio)
        img_bytes = fig_alloc.to_image(format='png', width=600, height=400, engine='kaleido')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        pdf.image(tmp_path, x=15, w=180)
        os.unlink(tmp_path)
    except Exception:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 8, '(Chart rendering unavailable)', new_x='LMARGIN', new_y='NEXT')

    # --- Page 2: Risk Analysis ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Risk Analysis', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)

    # Risk metrics table
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(100, 8, 'Metric', border=1, fill=True)
    pdf.cell(70, 8, 'Value', border=1, fill=True, align='C')
    pdf.ln()

    metrics_rows = [
        ('Daily Volatility', f"{risk_metrics['daily_volatility']*100:.4f}%"),
        ('Annual Volatility', f"{risk_metrics['annual_volatility']*100:.2f}%"),
        ('Sharpe Ratio', f"{sharpe:.2f}"),
        ('Max Drawdown', f"{max_drawdown*100:.2f}%"),
    ]
    pdf.set_font('Helvetica', '', 10)
    for label, value in metrics_rows:
        pdf.cell(100, 8, label, border=1)
        pdf.cell(70, 8, value, border=1, align='C')
        pdf.ln()

    pdf.ln(5)

    # Risk gauge chart
    try:
        fig_risk = MutualFundPortfolio.plot_risk_metrics(
            risk_metrics['annual_volatility'], risk_metrics['daily_volatility']
        )
        img_bytes = fig_risk.to_image(format='png', width=700, height=350, engine='kaleido')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        pdf.image(tmp_path, x=10, w=190)
        os.unlink(tmp_path)
    except Exception:
        pass

    # Correlation heatmap
    try:
        fig_corr = MutualFundPortfolio.plot_correlation_heatmap(risk_metrics['correlation_matrix'])
        img_bytes = fig_corr.to_image(format='png', width=600, height=500, engine='kaleido')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        pdf.image(tmp_path, x=15, w=180)
        os.unlink(tmp_path)
    except Exception:
        pass

    # --- Page 3: Monte Carlo Results ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Monte Carlo Simulation Results', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)

    # MC metrics table
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(80, 8, 'Metric', border=1, fill=True)
    pdf.cell(50, 8, '3 Months', border=1, fill=True, align='C')
    pdf.cell(50, 8, '6 Months', border=1, fill=True, align='C')
    pdf.ln()

    mc_rows = [
        ('Bottom 10%', results_3m['bottom_10'], results_6m['bottom_10']),
        ('Bottom 5%', results_3m['percentile_5'], results_6m['percentile_5']),
        ('Median', results_3m['median'], results_6m['median']),
        ('Mean', results_3m['mean'], results_6m['mean']),
        ('Top 20%', results_3m['top_20'], results_6m['top_20']),
        ('Top 5%', results_3m['percentile_95'], results_6m['percentile_95']),
    ]
    pdf.set_font('Helvetica', '', 10)
    for label, v3, v6 in mc_rows:
        pdf.cell(80, 8, label, border=1)
        pdf.cell(50, 8, f'{v3*100:.2f}%', border=1, align='C')
        pdf.cell(50, 8, f'{v6*100:.2f}%', border=1, align='C')
        pdf.ln()

    pdf.ln(5)

    # MC distribution chart (3M)
    try:
        fig_mc = MutualFundPortfolio.plot_monte_carlo_distribution(
            results_3m['final_returns'], results_3m['median'],
            results_3m['top_20'], results_3m['bottom_10']
        )
        fig_mc.update_layout(title="3-Month Return Distribution")
        img_bytes = fig_mc.to_image(format='png', width=700, height=400, engine='kaleido')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        pdf.image(tmp_path, x=10, w=190)
        os.unlink(tmp_path)
    except Exception:
        pass

    # Percentile comparison chart
    try:
        fig_comp = MutualFundPortfolio.plot_percentile_comparison(results_3m, results_6m)
        img_bytes = fig_comp.to_image(format='png', width=700, height=400, engine='kaleido')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        pdf.image(tmp_path, x=10, w=190)
        os.unlink(tmp_path)
    except Exception:
        pass

    return bytes(pdf.output())


def main():
    #"""Main Streamlit app"""

    # Header
    st.markdown('<h1 class="main-header">📊 Portfolio Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Get DB manager
    db = get_db_manager()

    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = MutualFundPortfolio()
        st.session_state.funds_list = []
        st.session_state.analysis_complete = False
        st.session_state.search_results = None

    # Sidebar - Portfolio Builder
    with st.sidebar:
        st.header("🎯 Portfolio Builder")

        # --- Search Section ---
        st.subheader("Search Assets")

        search_query = st.text_input(
            "Search by name, symbol, or ISIN:",
            placeholder="e.g., Axis Bluechip, RELIANCE, INE..."
        )

        asset_type_filter = st.multiselect(
            "Filter by type (optional):",
            options=['mutual_fund', 'stock'],
            format_func=lambda x: 'Mutual Fund' if x == 'mutual_fund' else 'Stock'
        )

        if st.button("🔍 Search", use_container_width=True):
            if search_query and len(search_query) >= 2:
                with st.spinner("Searching..."):
                    results = db.search_all_assets(
                        search_query,
                        asset_types=asset_type_filter if asset_type_filter else None,
                        limit=20
                    )
                    st.session_state.search_results = results
            else:
                st.warning("Enter at least 2 characters to search")

        # --- Search Results & Selection ---
        if st.session_state.search_results is not None and not st.session_state.search_results.empty:
            results_df = st.session_state.search_results

            # Build display options
            options = []
            isin_map = {}
            for _, row in results_df.iterrows():
                display_name = row.get('full_name') or row.get('name', '')
                symbol = row.get('symbol', '')
                asset_type = row.get('asset_type', '')
                isin = row.get('isin', '')
                type_label = 'MF' if asset_type == 'mutual_fund' else 'Stock'
                option_str = f"{display_name} | {symbol} | {type_label}"
                options.append(option_str)
                isin_map[option_str] = {
                    'isin': isin,
                    'display_name': display_name,
                    'symbol': symbol,
                    'asset_type': asset_type
                }

            selected = st.selectbox(
                f"Results ({len(options)} found):",
                options=options
            )

            allocation = st.slider("Allocation (%):", 0, 100, 25, 1)

            if st.button("➕ Add to Portfolio", use_container_width=True):
                if selected:
                    info = isin_map[selected]
                    # Check for duplicate ISINs
                    existing_isins = {f['isin'] for f in st.session_state.funds_list}
                    if info['isin'] in existing_isins:
                        st.warning("This asset is already in your portfolio")
                    else:
                        display_name = info['display_name']
                        try:
                            st.session_state.portfolio.add_fund(display_name, allocation)
                            st.session_state.funds_list.append({
                                'display_name': display_name,
                                'isin': info['isin'],
                                'symbol': info['symbol'],
                                'asset_type': info['asset_type'],
                                'allocation': allocation
                            })
                            st.success(f"Added {display_name[:40]}")
                        except Exception as e:
                            st.error(f"Error: {e}")
        elif st.session_state.search_results is not None:
            st.info("No results found. Try a different search term.")

        st.markdown("---")

        # --- Clear All ---
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.portfolio = MutualFundPortfolio()
            st.session_state.funds_list = []
            st.session_state.analysis_complete = False
            st.session_state.search_results = None
            st.rerun()

        st.markdown("---")

        # Current portfolio
        st.subheader("Current Portfolio")

        if st.session_state.funds_list:
            for idx, fund_info in enumerate(st.session_state.funds_list):
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    label = fund_info['display_name']
                    if len(label) > 25:
                        label = label[:22] + '...'
                    st.text(label)
                with col2:
                    st.text(f"{fund_info['allocation']}%")
                with col3:
                    if st.button("❌", key=f"del_{idx}"):
                        del st.session_state.funds_list[idx]
                        st.session_state.portfolio = MutualFundPortfolio()
                        for f in st.session_state.funds_list:
                            st.session_state.portfolio.add_fund(f['display_name'], f['allocation'])
                        st.session_state.analysis_complete = False
                        st.rerun()

            # Validate allocation
            is_valid, total = st.session_state.portfolio.validate_portfolio()

            if is_valid:
                st.success(f"✅ Total: {total:.1f}%")
            else:
                st.error(f"❌ Total: {total:.1f}% (must be 100%)")

            st.markdown("---")

            # Analysis parameters
            st.subheader("Analysis Settings")

            num_simulations = st.selectbox(
                "Monte Carlo Simulations:",
                [1000, 5000, 10000, 20000],
                index=2
            )

            st.markdown("---")

            # Run analysis button
            if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
                if is_valid:
                    st.session_state.analysis_complete = False
                    with st.spinner("Fetching data and running analysis..."):
                        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

                        progress_bar = st.progress(0)
                        total_funds = len(st.session_state.funds_list)

                        fetch_ok = True
                        for idx, fund_info in enumerate(st.session_state.funds_list):
                            result = st.session_state.portfolio.fetch_price_from_db(
                                fund_info['isin'],
                                fund_info['display_name'],
                                db,
                                start_date=start_date
                            )
                            if result is None:
                                st.error(f"No price data found for {fund_info['display_name']}")
                                fetch_ok = False
                            progress_bar.progress((idx + 1) / total_funds)

                        if fetch_ok:
                            # Calculate returns
                            st.session_state.portfolio.calculate_returns()

                            # Calculate risk
                            st.session_state.risk_metrics = st.session_state.portfolio.calculate_portfolio_risk()

                            # Run Monte Carlo
                            st.session_state.results_3m = st.session_state.portfolio.monte_carlo_simulation(
                                months=3,
                                num_simulations=num_simulations
                            )
                            st.session_state.results_6m = st.session_state.portfolio.monte_carlo_simulation(
                                months=6,
                                num_simulations=num_simulations
                            )

                            st.session_state.analysis_complete = True
                            st.success("Analysis complete!")
                            st.rerun()
                        else:
                            st.error("Could not fetch data for all assets. Check selections.")
                else:
                    st.error("Please ensure allocations total 100%")
        else:
            st.info("Search and add assets to build your portfolio")

    # Main content area
    if st.session_state.funds_list:
        # Portfolio Overview
        st.header("📈 Portfolio Overview")
        st.markdown(
            "This tool analyses your portfolio using historical price data and simulates "
            "thousands of possible future scenarios to show you the range of returns you "
            "might expect — from the worst case to the best case and everything in between. "
            "Think of it as a weather forecast for your investments: not a guarantee, but a "
            "well-informed view of what could happen."
        )

        # Portfolio allocation pie chart
        fig_allocation = MutualFundPortfolio.plot_portfolio_allocation(st.session_state.portfolio.portfolio)
        st.plotly_chart(fig_allocation, use_container_width=True)

        # # Portfolio summary (commented out)
        # st.subheader("Portfolio Summary")
        # total_funds = len(st.session_state.funds_list)
        # st.metric("Total Assets", total_funds)
        # if st.session_state.analysis_complete:
        #     returns_data = st.session_state.portfolio.returns_data
        #     st.metric(
        #         "Data Points",
        #         len(returns_data),
        #         help="Number of trading days in analysis"
        #     )
        #     st.metric(
        #         "Date Range",
        #         f"{returns_data.index[0].strftime('%Y-%m-%d')} to {returns_data.index[-1].strftime('%Y-%m-%d')}"
        #     )

        if st.session_state.analysis_complete:
            st.markdown("---")

            # NAV History
            st.header("📊 Historical Performance")
            fig_nav =MutualFundPortfolio.plot_nav_history(st.session_state.portfolio.nav_data)
            st.plotly_chart(fig_nav, use_container_width=True)

            st.markdown("---")

            # Risk Analysis
            st.header("⚠️ Portfolio Risk Analysis")

            risk_metrics = st.session_state.risk_metrics

            # Calculate Sharpe and Max Drawdown
            returns = st.session_state.portfolio.returns_data
            weights = np.array([st.session_state.portfolio.portfolio[f] for f in returns.columns])
            portfolio_returns = (returns * weights).sum(axis=1)
            sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
            max_drawdown = ((portfolio_returns + 1).cumprod().cummax() - (portfolio_returns + 1).cumprod()).max()

            # Annual Volatility gauge
            fig_risk = MutualFundPortfolio.plot_risk_metrics(
                risk_metrics['annual_volatility']
            )
            st.plotly_chart(fig_risk, use_container_width=False)

            st.markdown(
                "**Annual Volatility** measures the degree of variation in your portfolio's "
                "returns over a year. A lower value indicates more stable, predictable returns, "
                "while a higher value signals larger price swings. "
                "Generally, below 15% is considered low risk, 15-25% moderate, and above 25% high risk."
            )

            st.markdown("")

            # Risk stats table
            st.subheader("Risk Statistics")
            risk_table = pd.DataFrame({
                'Metric': ['Annual Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                'Value': [
                    f"{risk_metrics['annual_volatility']*100:.2f}%",
                    f"{sharpe:.2f}",
                    f"{max_drawdown*100:.2f}%"
                ],
                'Description': [
                    'Annualized standard deviation of portfolio returns. Lower means more stable.',
                    'Risk-adjusted return (return per unit of risk). Above 1.0 is excellent, below 0.5 is poor.',
                    'Largest peak-to-trough decline in portfolio value. Indicates worst-case historical loss.'
                ]
            })
            st.dataframe(risk_table, use_container_width=True, hide_index=True)

            # Correlation heatmap
            st.subheader("Fund Correlations")
            fig_corr = MutualFundPortfolio.plot_correlation_heatmap(risk_metrics['correlation_matrix'])
            st.plotly_chart(fig_corr, use_container_width=False)

            st.markdown("---")

            # Monte Carlo Results
            st.header("🎲 Monte Carlo Simulation Results")
            st.markdown(
                "We ran thousands of simulated futures for your portfolio based on how it has "
                "behaved in the past. The results below show the range of outcomes you could "
                "see — helping you understand not just what's likely, but also what's possible "
                "in a good or bad scenario."
            )

            # Tabs for different time horizons
            tab1, tab2, tab3 = st.tabs(["3-Month Forecast", "6-Month Forecast", "Comparison"])

            with tab1:
                st.subheader("3-Month Outlook")

                results = st.session_state.results_3m

                # Key metrics
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric(
                        "Bottom 10%",
                        f"{results['bottom_10']*100:.2f}%",
                        delta=None,
                        delta_color="inverse"
                    )

                with col2:
                    st.metric(
                        "Median (50%)",
                        f"{results['median']*100:.2f}%"
                    )

                with col3:
                    st.metric(
                        "Top 20%",
                        f"{results['top_20']*100:.2f}%",
                        delta=None,
                        delta_color="normal"
                    )

                with col4:
                    st.metric(
                        "Mean",
                        f"{results['mean']*100:.2f}%"
                    )

                with col5:
                    st.metric(
                        "Std Dev",
                        f"{results['std']*100:.2f}%"
                    )

                # Distribution plot
                fig_dist_3m = MutualFundPortfolio.plot_monte_carlo_distribution(
                    results['final_returns'],
                    results['median'],
                    results['top_20'],
                    results['bottom_10']
                )
                st.plotly_chart(fig_dist_3m, use_container_width=True)

                # Simulation paths
                fig_paths_3m = MutualFundPortfolio.plot_simulation_paths(results['simulations'], num_paths=100)
                st.plotly_chart(fig_paths_3m, use_container_width=True)

            with tab2:
                st.subheader("6-Month Outlook")

                results = st.session_state.results_6m

                # Key metrics
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric(
                        "Bottom 10%",
                        f"{results['bottom_10']*100:.2f}%",
                        delta=None,
                        delta_color="inverse"
                    )

                with col2:
                    st.metric(
                        "Median (50%)",
                        f"{results['median']*100:.2f}%"
                    )

                with col3:
                    st.metric(
                        "Top 20%",
                        f"{results['top_20']*100:.2f}%",
                        delta=None,
                        delta_color="normal"
                    )

                with col4:
                    st.metric(
                        "Mean",
                        f"{results['mean']*100:.2f}%"
                    )

                with col5:
                    st.metric(
                        "Std Dev",
                        f"{results['std']*100:.2f}%"
                    )

                # Distribution plot
                fig_dist_6m = MutualFundPortfolio.plot_monte_carlo_distribution(
                    results['final_returns'],
                    results['median'],
                    results['top_20'],
                    results['bottom_10']
                )
                st.plotly_chart(fig_dist_6m, use_container_width=True)

                # Simulation paths
                fig_paths_6m = MutualFundPortfolio.plot_simulation_paths(results['simulations'], num_paths=100)
                st.plotly_chart(fig_paths_6m, use_container_width=True)

            with tab3:
                st.subheader("Time Horizon Comparison")
                st.markdown("Side-by-side view of how your portfolio's expected outcomes change when you stay invested for 3 months versus 6 months.")

                # Comparison chart
                fig_comparison = MutualFundPortfolio.plot_percentile_comparison(
                    st.session_state.results_3m,
                    st.session_state.results_6m
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

                # Detailed comparison table
                st.subheader("Detailed Statistics")
                st.markdown("A breakdown of key return scenarios across both time horizons, from the worst 5% to the best 5% of outcomes.")

                comparison_df = pd.DataFrame({
                    'Metric': ['Bottom 10%', 'Bottom 5%', 'Median', 'Mean', 'Top 20%', 'Top 5%'],
                    '3 Months': [
                        f"{st.session_state.results_3m['bottom_10']*100:.2f}%",
                        f"{st.session_state.results_3m['percentile_5']*100:.2f}%",
                        f"{st.session_state.results_3m['median']*100:.2f}%",
                        f"{st.session_state.results_3m['mean']*100:.2f}%",
                        f"{st.session_state.results_3m['top_20']*100:.2f}%",
                        f"{st.session_state.results_3m['percentile_95']*100:.2f}%"
                    ],
                    '6 Months': [
                        f"{st.session_state.results_6m['bottom_10']*100:.2f}%",
                        f"{st.session_state.results_6m['percentile_5']*100:.2f}%",
                        f"{st.session_state.results_6m['median']*100:.2f}%",
                        f"{st.session_state.results_6m['mean']*100:.2f}%",
                        f"{st.session_state.results_6m['top_20']*100:.2f}%",
                        f"{st.session_state.results_6m['percentile_95']*100:.2f}%"
                    ]
                })

                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Download results
            st.header("💾 Export Results")

            # Prepare export data
            summary = {
                'Portfolio': list(st.session_state.portfolio.portfolio.keys()),
                'Allocation (%)': [v*100 for v in st.session_state.portfolio.portfolio.values()],
                'Annual Volatility (%)': [risk_metrics['annual_volatility']*100] * len(st.session_state.portfolio.portfolio),
                '3M Median Return (%)': [st.session_state.results_3m['median']*100] * len(st.session_state.portfolio.portfolio),
                '6M Median Return (%)': [st.session_state.results_6m['median']*100] * len(st.session_state.portfolio.portfolio),
            }
            summary_df = pd.DataFrame(summary)
            csv_summary = summary_df.to_csv(index=False)

            detailed_results = {
                'Metric': [
                    'Daily Volatility (%)', 'Annual Volatility (%)',
                    'Sharpe Ratio', 'Max Drawdown (%)',
                    '3M Bottom 10% (%)', '3M Median (%)', '3M Top 20% (%)', '3M Mean (%)',
                    '6M Bottom 10% (%)', '6M Median (%)', '6M Top 20% (%)', '6M Mean (%)',
                ],
                'Value': [
                    f"{risk_metrics['daily_volatility']*100:.4f}",
                    f"{risk_metrics['annual_volatility']*100:.2f}",
                    f"{sharpe:.2f}", f"{max_drawdown*100:.2f}",
                    f"{st.session_state.results_3m['bottom_10']*100:.2f}",
                    f"{st.session_state.results_3m['median']*100:.2f}",
                    f"{st.session_state.results_3m['top_20']*100:.2f}",
                    f"{st.session_state.results_3m['mean']*100:.2f}",
                    f"{st.session_state.results_6m['bottom_10']*100:.2f}",
                    f"{st.session_state.results_6m['median']*100:.2f}",
                    f"{st.session_state.results_6m['top_20']*100:.2f}",
                    f"{st.session_state.results_6m['mean']*100:.2f}",
                ]
            }
            detailed_df = pd.DataFrame(detailed_results)
            csv_detailed = detailed_df.to_csv(index=False)

            try:
                pdf_bytes = generate_pdf_report(
                    st.session_state.portfolio, risk_metrics,
                    st.session_state.results_3m, st.session_state.results_6m,
                    sharpe, max_drawdown
                )
            except Exception:
                pdf_bytes = None

            # Download buttons — always available
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="📥 Portfolio Summary (CSV)",
                    data=csv_summary,
                    file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv", use_container_width=True
                )
            with col2:
                st.download_button(
                    label="📥 Detailed Results (CSV)",
                    data=csv_detailed,
                    file_name=f"detailed_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv", use_container_width=True
                )
            with col3:
                if pdf_bytes:
                    st.download_button(
                        label="📥 Full Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf", use_container_width=True
                    )
                else:
                    st.error("PDF generation failed")

            # Optional email collection
            if not st.session_state.get('email_verified'):
                st.markdown("")
                st.info(
                    "**Get free personalised advice!** Share your email to receive "
                    "one-on-one portfolio guidance from a SEBI-registered Mutual Fund "
                    "Distributor — completely free of cost."
                )
                email_input = st.text_input(
                    "Your email (optional):",
                    placeholder="you@example.com",
                    key="email_input_field"
                )
                if st.button("Submit", use_container_width=True):
                    if not email_input:
                        st.warning("Please enter your email address.")
                    elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email_input.strip()):
                        st.error("Please enter a valid email address (e.g., name@example.com).")
                    else:
                        success, msg = db.insert_user_email(email_input)
                        if success:
                            st.session_state.email_verified = True
                            st.success("Thank you! A SEBI-registered advisor will reach out to you shortly.")
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.success("Thank you for sharing your email! An advisor will be in touch.")

            st.markdown("---")
            st.header("📖 Understanding Your Results")

            with st.expander("🎯 What do the percentiles mean?"):
                st.markdown("""
                **Bottom 10th Percentile**: There's a 90% chance your returns will be better than this value.
                This represents a pessimistic but realistic worst-case scenario.

                **Median (50th Percentile)**: Half of the simulations resulted in returns better than this,
                and half resulted in worse returns. This is the most likely outcome.

                **Top 20th Percentile**: There's a 20% chance your returns will exceed this value.
                This represents an optimistic but achievable scenario.
                """)

            with st.expander("⚠️ How to interpret volatility?"):
                st.markdown("""
                **Daily Volatility**: Day-to-day price fluctuations. Lower is generally better for risk-averse investors.

                **Annual Volatility**:
                - < 15%: Low risk (conservative portfolio)
                - 15-25%: Moderate risk (balanced portfolio)
                - > 25%: High risk (aggressive portfolio)

                Higher volatility means more uncertainty but potentially higher returns.
                """)

            with st.expander("🔗 What does correlation tell you?"):
                st.markdown("""
                **Correlation Matrix**: Shows how funds move together.

                - **+1.0**: Perfect positive correlation (move together)
                - **0.0**: No correlation (move independently)
                - **-1.0**: Perfect negative correlation (move opposite)

                **Diversification Tip**: Look for funds with lower correlations (< 0.7) to reduce portfolio risk.
                """)

            with st.expander("📊 Monte Carlo Simulation explained"):
                st.markdown("""
                **What it does**: Runs thousands of scenarios based on historical data to project potential future outcomes.

                **How to use it**:
                1. Focus on the median for the most likely outcome
                2. Use bottom 10% for risk assessment and planning
                3. Use top 20% for upside potential
                4. Compare 3-month vs 6-month to understand time horizon impact

                **Important**: Past performance doesn't guarantee future results. These are probabilistic projections.
                """)

            with st.expander("✅ Investment recommendations"):
                st.markdown("### Based on your portfolio analysis:")

                if risk_metrics['annual_volatility'] > 0.30:
                    st.warning("🔴 **High Risk Detected**: Your portfolio has high volatility (>30%). Consider adding more stable, low-correlation funds.")
                elif risk_metrics['annual_volatility'] < 0.10:
                    st.info("🟢 **Conservative Portfolio**: Very low volatility. You might consider adding growth-oriented funds for better returns.")
                else:
                    st.success("🟡 **Balanced Risk**: Your portfolio has moderate volatility, suitable for most investors.")

                avg_correlation = risk_metrics['correlation_matrix'].values[np.triu_indices_from(risk_metrics['correlation_matrix'].values, k=1)].mean()

                if avg_correlation > 0.8:
                    st.warning("🔴 **Low Diversification**: Your funds are highly correlated. Consider adding funds from different sectors/asset classes.")
                elif avg_correlation < 0.5:
                    st.success("🟢 **Well Diversified**: Good diversification with low average correlation between funds.")
                else:
                    st.info("🟡 **Moderate Diversification**: Decent diversification, but there's room for improvement.")

                if st.session_state.results_6m['median'] < 0:
                    st.error("🔴 **Negative Expected Returns**: The median projection is negative. Review your fund selection.")
                elif st.session_state.results_6m['median'] > 0.10:
                    st.success("🟢 **Strong Growth Potential**: High expected returns, but monitor risk carefully.")

                if sharpe < 0.5:
                    st.warning("🔴 **Low Risk-Adjusted Returns**: Consider rebalancing for better risk-return tradeoff.")
                elif sharpe > 1.0:
                    st.success("🟢 **Excellent Risk-Adjusted Returns**: Strong performance relative to risk taken.")

    else:
        # Welcome screen
        st.info("""
        ### 👋 Welcome to the Portfolio Analyzer!

        Search for stocks and mutual funds in the sidebar to build your portfolio.
        """)
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Made with ❤️ using Streamlit |
        <a href='https://github.com' target='_blank'>GitHub</a> |
        <a href='https://docs.streamlit.io' target='_blank'>Documentation</a></p>
        <p style='font-size: 0.8em;'>⚠️ Disclaimer: This tool is for educational purposes only.
        Not financial advice. Consult a professional advisor before investing.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
