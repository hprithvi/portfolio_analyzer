#I'll create an interactive Streamlit app with graphical inputs and outputs.

#```python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# Set page config
st.set_page_config(
    page_title="Mutual Fund Portfolio Analyzer",
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
    
    def fetch_nav_data(self, fund_symbol, start_date=None, end_date=None):
        #"""Fetch historical NAV data for a mutual fund"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
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
    #"""Create correlation heatmap"""
        fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
        fig.update_layout(
        title="Fund Correlation Matrix",
        height=500,
        xaxis=dict(tickangle=-45)
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
        title="Normalized NAV Performance (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Normalized NAV",
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


    def plot_risk_metrics(annual_volatility, daily_volatility):
    #"""Create gauge charts for risk metrics"""
        fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
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
    ), row=1, col=1)
    
        fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=daily_volatility * 100,
        title={'text': "Daily Volatility (%)"},
        gauge={
            'axis': {'range': [0, 3]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 1], 'color': "lightgreen"},
                {'range': [1, 2], 'color': "yellow"},
                {'range': [2, 3], 'color': "lightcoral"}
            ]
        }
    ), row=1, col=2)
    
        fig.update_layout(height=400)
    
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


def main():
    #"""Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Mutual Fund Portfolio Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = MutualFundPortfolio()
        st.session_state.funds_list = []
        st.session_state.analysis_complete = False
    
    # Sidebar - Portfolio Builder
    with st.sidebar:
        st.header("🎯 Portfolio Builder")
        
        # Fund type selection
        fund_type = st.radio(
            "Select Fund Type:",
            ["Indian Mutual Funds (AMFI)", "ETFs/International Funds"]
        )
        
        st.markdown("---")
        
        # Add fund section
        st.subheader("Add Fund")
        
        if fund_type == "Indian Mutual Funds (AMFI)":
            fund_input = st.text_input("AMFI Scheme Code:", placeholder="e.g., 120503")
            st.caption("Find scheme codes at [AMFI India](https://www.amfiindia.com)")
        else:
            fund_input = st.text_input("Fund Symbol/Ticker:", placeholder="e.g., SPY, QQQ")
            st.caption("Use Yahoo Finance ticker symbols")
        
        allocation = st.slider("Allocation (%):", 0, 100, 25, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Add Fund", use_container_width=True):
                if fund_input:
                    try:
                        st.session_state.portfolio.add_fund(fund_input, allocation)
                        st.session_state.funds_list.append({
                            'fund': fund_input,
                            'allocation': allocation,
                            'type': fund_type
                        })
                        st.success(f"Added {fund_input}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please enter a fund code/symbol")
        
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.portfolio = MutualFundPortfolio()
                st.session_state.funds_list = []
                st.session_state.analysis_complete = False
                st.rerun()
        
        st.markdown("---")
        
        # Current portfolio
        st.subheader("Current Portfolio")
        
        if st.session_state.funds_list:
            for idx, fund_info in enumerate(st.session_state.funds_list):
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.text(fund_info['fund'])
                with col2:
                    st.text(f"{fund_info['allocation']}%")
                with col3:
                    if st.button("❌", key=f"del_{idx}"):
                        del st.session_state.funds_list[idx]
                        st.session_state.portfolio = MutualFundPortfolio()
                        for f in st.session_state.funds_list:
                            st.session_state.portfolio.add_fund(f['fund'], f['allocation'])
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
            
            lookback_years = st.slider(
                "Historical Data (years):",
                1, 5, 3
            )
            
            st.markdown("---")
            
            # Run analysis button
            if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
                if is_valid:
                    st.session_state.analysis_complete = False
                    with st.spinner("Fetching data and running analysis..."):
                        # Fetch data
                        start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
                        
                        progress_bar = st.progress(0)
                        total_funds = len(st.session_state.funds_list)
                        
                        for idx, fund_info in enumerate(st.session_state.funds_list):
                            if fund_info['type'] == "Indian Mutual Funds (AMFI)":
                                st.session_state.portfolio.fetch_indian_mf_data(
                                    fund_info['fund'], 
                                    start_date=start_date
                                )
                            else:
                                st.session_state.portfolio.fetch_nav_data(
                                    fund_info['fund'], 
                                    start_date=start_date
                                )
                            progress_bar.progress((idx + 1) / total_funds)
                        
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
                    st.error("Please ensure allocations total 100%")
        else:
            st.info("Add funds to start building your portfolio")
    
    # Main content area
    if st.session_state.funds_list:
        # Portfolio Overview
        st.header("📈 Portfolio Overview")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Portfolio allocation pie chart
            fig_allocation = plot_portfolio_allocation(st.session_state.portfolio.portfolio)
            st.plotly_chart(fig_allocation, use_container_width=True)
        
        with col2:
            # Portfolio summary
            st.subheader("Portfolio Summary")
            
            total_funds = len(st.session_state.funds_list)
            st.metric("Total Funds", total_funds)
            
            if st.session_state.analysis_complete:
                returns_data = st.session_state.portfolio.returns_data
                st.metric(
                    "Data Points", 
                    len(returns_data),
                    help="Number of trading days in analysis"
                )
                st.metric(
                    "Date Range",
                    f"{returns_data.index[0].strftime('%Y-%m-%d')} to {returns_data.index[-1].strftime('%Y-%m-%d')}"
                )
        
        if st.session_state.analysis_complete:
            st.markdown("---")
            
            # NAV History
            st.header("📊 Historical Performance")
            fig_nav = plot_nav_history(st.session_state.portfolio.nav_data)
            st.plotly_chart(fig_nav, use_container_width=True)
            
            st.markdown("---")
            
            # Risk Analysis
            st.header("⚠️ Portfolio Risk Analysis")
            
            risk_metrics = st.session_state.risk_metrics
            
            # Risk gauges
            fig_risk = plot_risk_metrics(
                risk_metrics['annual_volatility'],
                risk_metrics['daily_volatility']
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Daily Volatility",
                    f"{risk_metrics['daily_volatility']*100:.4f}%"
                )
            
            with col2:
                st.metric(
                    "Annual Volatility",
                    f"{risk_metrics['annual_volatility']*100:.2f}%"
                )
            
            with col3:
                # Calculate Sharpe ratio (assuming 0% risk-free rate)
                returns = st.session_state.portfolio.returns_data
                weights = np.array([st.session_state.portfolio.portfolio[f] for f in returns.columns])
                portfolio_returns = (returns * weights).sum(axis=1)
                sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            with col4:
                max_drawdown = ((portfolio_returns + 1).cumprod().cummax() - (portfolio_returns + 1).cumprod()).max()
                st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
            
            # Correlation heatmap
            st.subheader("Fund Correlations")
            fig_corr = plot_correlation_heatmap(risk_metrics['correlation_matrix'])
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown("---")
            
            # Monte Carlo Results
            st.header("🎲 Monte Carlo Simulation Results")
            
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
                fig_dist_3m = plot_monte_carlo_distribution(
                    results['final_returns'],
                    results['median'],
                    results['top_20'],
                    results['bottom_10']
                )
                st.plotly_chart(fig_dist_3m, use_container_width=True)
                
                # Simulation paths
                fig_paths_3m = plot_simulation_paths(results['simulations'], num_paths=100)
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
                fig_dist_6m = plot_monte_carlo_distribution(
                    results['final_returns'],
                    results['median'],
                    results['top_20'],
                    results['bottom_10']
                )
                st.plotly_chart(fig_dist_6m, use_container_width=True)
                
                # Simulation paths
                fig_paths_6m = plot_simulation_paths(results['simulations'], num_paths=100)
                st.plotly_chart(fig_paths_6m, use_container_width=True)
            
            with tab3:
                st.subheader("Time Horizon Comparison")
                
                # Comparison chart
                fig_comparison = plot_percentile_comparison(
                    st.session_state.results_3m,
                    st.session_state.results_6m
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Detailed comparison table
                st.subheader("Detailed Statistics")
                
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
            
        col1, col2 = st.columns(2)
            
        with col1:
            # Create summary report
            summary = {'Portfolio': list(st.session_state.portfolio.portfolio.keys()),
                    'Allocation (%)': [v*100 for v in st.session_state.portfolio.portfolio.values()],
                    'Annual Volatility (%)': [risk_metrics['annual_volatility']*100] * len(st.session_state.portfolio.portfolio),
                    '3M Median Return (%)': [st.session_state.results_3m['median']*100] * len(st.session_state.portfolio.portfolio),
                    '6M Median Return (%)': [st.session_state.results_6m['median']*100] * len(st.session_state.portfolio.portfolio),
                }
                
                summary_df = pd.DataFrame(summary)
                
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Portfolio Summary (CSV)",
                    data=csv,
                    file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Create detailed results
                detailed_results = {
                    'Metric': [
                        'Daily Volatility (%)',
                        'Annual Volatility (%)',
                        'Sharpe Ratio',
                        'Max Drawdown (%)',
                        '3M Bottom 10% (%)',
                        '3M Median (%)',
                        '3M Top 20% (%)',
                        '3M Mean (%)',
                        '6M Bottom 10% (%)',
                        '6M Median (%)',
                        '6M Top 20% (%)',
                        '6M Mean (%)',
                    ],
                    'Value': [
                        f"{risk_metrics['daily_volatility']*100:.4f}",
                        f"{risk_metrics['annual_volatility']*100:.2f}",
                        f"{sharpe:.2f}",
                        f"{max_drawdown*100:.2f}",
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
                
                st.download_button(
                    label="📥 Download Detailed Results (CSV)",
                    data=csv_detailed,
                    file_name=f"detailed_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Interpretation guide
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
                # Dynamic recommendations based on results
                st.markdown("### Based on your portfolio analysis:")
                
                if risk_metrics['annual_volatility'] > 0.30:
                    st.warning("🔴 **High Risk Detected**: Your portfolio has high volatility (>30%). Consider adding more stable, low-correlation funds.")
                elif risk_metrics['annual_volatility'] < 0.10:
                    st.info("🟢 **Conservative Portfolio**: Very low volatility. You might consider adding growth-oriented funds for better returns.")
                else:
                    st.success("🟡 **Balanced Risk**: Your portfolio has moderate volatility, suitable for most investors.")
                
                # Check correlation
                avg_correlation = risk_metrics['correlation_matrix'].values[np.triu_indices_from(risk_metrics['correlation_matrix'].values, k=1)].mean()
                
                if avg_correlation > 0.8:
                    st.warning("🔴 **Low Diversification**: Your funds are highly correlated. Consider adding funds from different sectors/asset classes.")
                elif avg_correlation < 0.5:
                    st.success("🟢 **Well Diversified**: Good diversification with low average correlation between funds.")
                else:
                    st.info("🟡 **Moderate Diversification**: Decent diversification, but there's room for improvement.")
                
                # Check expected returns
                if st.session_state.results_6m['median'] < 0:
                    st.error("🔴 **Negative Expected Returns**: The median projection is negative. Review your fund selection.")
                elif st.session_state.results_6m['median'] > 0.10:
                    st.success("🟢 **Strong Growth Potential**: High expected returns, but monitor risk carefully.")
                
                # Sharpe ratio assessment
                if sharpe < 0.5:
                    st.warning("🔴 **Low Risk-Adjusted Returns**: Consider rebalancing for better risk-return tradeoff.")
                elif sharpe > 1.0:
                    st.success("🟢 **Excellent Risk-Adjusted Returns**: Strong performance relative to risk taken.")
    
    else:
        # Welcome screen
        st.info("""
        ### 👋 Welcome to the Mutual Fund Portfolio Analyzer!""")
