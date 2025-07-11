import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PapAIstra - AI Investment Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .recommendation-buy {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .recommendation-hold {
        background: linear-gradient(135deg, #f7931e 0%, #ffcc02 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .recommendation-sell {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, symbol, period="1y"):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            info = ticker.info
            
            if data.empty:
                return None
                
            return {
                'price_data': data,
                'company_info': info,
                'current_price': float(data['Close'].iloc[-1]),
                'daily_change': float(((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100) if len(data) > 1 else 0,
                'volume': int(data['Volume'].iloc[-1]) if not data['Volume'].empty else 0
            }
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, price_data):
        try:
            close = price_data['Close']
            
            # Simple Moving Averages
            sma_20 = close.rolling(window=20).mean()
            sma_50 = close.rolling(window=50).mean()
            sma_200 = close.rolling(window=200).mean()
            
            # RSI Calculation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            
            # Bollinger Bands
            bb_middle = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            return {
                'sma_20': float(sma_20.iloc[-1]) if not sma_20.empty else None,
                'sma_50': float(sma_50.iloc[-1]) if not sma_50.empty else None,
                'sma_200': float(sma_200.iloc[-1]) if not sma_200.empty else None,
                'rsi': float(rsi.iloc[-1]) if not rsi.empty else None,
                'macd': float(macd.iloc[-1]) if not macd.empty else None,
                'macd_signal': float(macd_signal.iloc[-1]) if not macd_signal.empty else None,
                'bb_upper': float(bb_upper.iloc[-1]) if not bb_upper.empty else None,
                'bb_lower': float(bb_lower.iloc[-1]) if not bb_lower.empty else None,
                'bb_middle': float(bb_middle.iloc[-1]) if not bb_middle.empty else None
            }
        except Exception as e:
            return {}
    
    def calculate_fundamental_metrics(self, company_info):
        try:
            return {
                'pe_ratio': company_info.get('trailingPE', 'N/A'),
                'forward_pe': company_info.get('forwardPE', 'N/A'),
                'peg_ratio': company_info.get('pegRatio', 'N/A'),
                'price_to_book': company_info.get('priceToBook', 'N/A'),
                'price_to_sales': company_info.get('priceToSalesTrailing12Months', 'N/A'),
                'debt_to_equity': company_info.get('debtToEquity', 'N/A'),
                'return_on_equity': company_info.get('returnOnEquity', 'N/A'),
                'return_on_assets': company_info.get('returnOnAssets', 'N/A'),
                'revenue_growth': company_info.get('revenueGrowth', 'N/A'),
                'earnings_growth': company_info.get('earningsGrowth', 'N/A'),
                'market_cap': company_info.get('marketCap', 'N/A'),
                'beta': company_info.get('beta', 'N/A'),
                'dividend_yield': company_info.get('dividendYield', 'N/A'),
                'sector': company_info.get('sector', 'N/A'),
                'industry': company_info.get('industry', 'N/A')
            }
        except Exception:
            return {}
    
    def generate_ai_recommendation(self, symbol, technical_data, fundamental_data, price_data):
        try:
            # Technical Score
            technical_score = 0.5
            
            # RSI Analysis
            rsi = technical_data.get('rsi')
            if rsi:
                if 30 <= rsi <= 70:
                    technical_score += 0.1
                elif rsi < 30:
                    technical_score += 0.2  # Oversold
                elif rsi > 70:
                    technical_score -= 0.2  # Overbought
            
            # Moving Average Trend
            sma_20 = technical_data.get('sma_20')
            sma_50 = technical_data.get('sma_50')
            current_price = float(price_data['Close'].iloc[-1])
            
            if sma_20 and sma_50:
                if sma_20 > sma_50 and current_price > sma_20:
                    technical_score += 0.2  # Uptrend
                elif sma_20 < sma_50 and current_price < sma_20:
                    technical_score -= 0.2  # Downtrend
            
            # MACD Signal
            macd = technical_data.get('macd')
            macd_signal = technical_data.get('macd_signal')
            if macd and macd_signal:
                if macd > macd_signal:
                    technical_score += 0.1  # Bullish
                else:
                    technical_score -= 0.1  # Bearish
            
            # Fundamental Score
            fundamental_score = 0.5
            
            pe_ratio = fundamental_data.get('pe_ratio')
            if pe_ratio and isinstance(pe_ratio, (int, float)):
                if 5 <= pe_ratio <= 20:
                    fundamental_score += 0.15
                elif pe_ratio > 40:
                    fundamental_score -= 0.15
            
            roe = fundamental_data.get('return_on_equity')
            if roe and isinstance(roe, (int, float)):
                if roe > 0.15:
                    fundamental_score += 0.1
                elif roe < 0:
                    fundamental_score -= 0.2
            
            revenue_growth = fundamental_data.get('revenue_growth')
            if revenue_growth and isinstance(revenue_growth, (int, float)):
                if revenue_growth > 0.1:
                    fundamental_score += 0.15
                elif revenue_growth < -0.05:
                    fundamental_score -= 0.15
            
            debt_to_equity = fundamental_data.get('debt_to_equity')
            if debt_to_equity and isinstance(debt_to_equity, (int, float)):
                if debt_to_equity < 0.5:
                    fundamental_score += 0.1
                elif debt_to_equity > 2:
                    fundamental_score -= 0.1
            
            # Overall Score
            overall_score = (technical_score * 0.6 + fundamental_score * 0.4)
            overall_score = max(0, min(1, overall_score))
            
            # Generate Recommendation
            if overall_score >= 0.75:
                recommendation = "STRONG BUY"
                confidence = "High"
            elif overall_score >= 0.6:
                recommendation = "BUY"
                confidence = "Medium-High"
            elif overall_score >= 0.4:
                recommendation = "HOLD"
                confidence = "Medium"
            elif overall_score >= 0.25:
                recommendation = "SELL"
                confidence = "Medium-High"
            else:
                recommendation = "STRONG SELL"
                confidence = "High"
            
            # Price Targets
            price_targets = {
                'current': round(current_price, 2),
                'target_high': round(current_price * 1.15, 2),
                'target_low': round(current_price * 0.85, 2),
                'stop_loss': round(current_price * 0.9, 2)
            }
            
            # Key Insights
            insights = []
            if rsi and rsi > 70:
                insights.append(f"RSI at {rsi:.1f} indicates potential overbought condition")
            elif rsi and rsi < 30:
                insights.append(f"RSI at {rsi:.1f} suggests potential oversold opportunity")
            
            if pe_ratio and isinstance(pe_ratio, (int, float)):
                if pe_ratio < 15:
                    insights.append(f"P/E ratio of {pe_ratio:.1f} suggests potential undervaluation")
                elif pe_ratio > 30:
                    insights.append(f"High P/E ratio of {pe_ratio:.1f} indicates growth expectations or overvaluation")
            
            if revenue_growth and isinstance(revenue_growth, (int, float)) and revenue_growth > 0.2:
                insights.append(f"Strong revenue growth of {revenue_growth*100:.1f}% shows business momentum")
            
            if not insights:
                insights.append("Comprehensive analysis completed - review metrics for detailed assessment")
            
            return {
                'symbol': symbol,
                'recommendation': recommendation,
                'confidence': confidence,
                'overall_score': round(overall_score, 3),
                'technical_score': round(technical_score, 3),
                'fundamental_score': round(fundamental_score, 3),
                'price_targets': price_targets,
                'key_insights': insights,
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'recommendation': 'HOLD',
                'confidence': 'Low',
                'overall_score': 0.5,
                'technical_score': 0.5,
                'fundamental_score': 0.5,
                'price_targets': {'current': 0, 'target_high': 0, 'target_low': 0, 'stop_loss': 0},
                'key_insights': ['Analysis completed with limited data'],
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

class PortfolioManager:
    def calculate_portfolio_metrics(self, holdings, current_prices):
        try:
            total_value = 0
            total_cost = 0
            positions = []
            
            for symbol, holding in holdings.items():
                if symbol in current_prices:
                    shares = holding['shares']
                    cost_basis = holding['cost_basis']
                    current_price = current_prices[symbol]
                    
                    position_value = shares * current_price
                    position_cost = shares * cost_basis
                    gain_loss = position_value - position_cost
                    gain_loss_pct = (gain_loss / position_cost) * 100 if position_cost > 0 else 0
                    
                    positions.append({
                        'symbol': symbol,
                        'shares': shares,
                        'cost_basis': cost_basis,
                        'current_price': current_price,
                        'position_value': position_value,
                        'position_cost': position_cost,
                        'gain_loss': gain_loss,
                        'gain_loss_pct': gain_loss_pct
                    })
                    
                    total_value += position_value
                    total_cost += position_cost
            
            total_return = ((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0
            
            return {
                'positions': positions,
                'total_value': total_value,
                'total_cost': total_cost,
                'total_return': total_return,
                'total_gain_loss': total_value - total_cost
            }
        except Exception:
            return None

def main():
    # Initialize
    analyzer = StockAnalyzer()
    portfolio_manager = PortfolioManager()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 PapAIstra</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Investment Analysis Platform</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        user_id = st.text_input("User ID", value="demo_user")
        
        st.subheader("Analysis Settings")
        analysis_period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        
        st.subheader("Features")
        show_technical = st.checkbox("Technical Analysis", value=True)
        show_fundamental = st.checkbox("Fundamental Analysis", value=True)
        show_ai_recommendation = st.checkbox("AI Recommendations", value=True)
        
        st.subheader("Risk Settings")
        risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1)
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Analysis", "💼 Portfolio", "🌍 Market Overview", "📈 Performance"])
    
    # Tab 1: Stock Analysis
    with tab1:
        st.header("📊 Individual Stock Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbol_input = st.text_input(
                "Enter Stock Symbol", 
                placeholder="e.g., AAPL, TSLA, MSFT, GOOGL",
                help="Enter a valid stock ticker symbol"
            ).upper().strip()
        
        with col2:
            if st.button("🔍 Analyze Stock", type="primary", use_container_width=True):
                if symbol_input:
                    st.session_state.current_symbol = symbol_input
        
        if symbol_input and symbol_input == st.session_state.get('current_symbol', ''):
            with st.spinner(f"🔄 Analyzing {symbol_input}..."):
                stock_data = analyzer.get_stock_data(symbol_input, period=analysis_period)
                
                if stock_data:
                    # Current Price Display
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "💰 Current Price", 
                            f"${stock_data['current_price']:.2f}"
                        )
                    
                    with col2:
                        change = stock_data['daily_change']
                        st.metric(
                            "📈 Daily Change", 
                            f"{change:.2f}%",
                            delta=f"{change:.2f}%"
                        )
                    
                    with col3:
                        market_cap = stock_data['company_info'].get('marketCap', 0)
                        if market_cap:
                            if market_cap >= 1e12:
                                cap_display = f"${market_cap/1e12:.2f}T"
                            elif market_cap >= 1e9:
                                cap_display = f"${market_cap/1e9:.2f}B"
                            else:
                                cap_display = f"${market_cap/1e6:.2f}M"
                        else:
                            cap_display = "N/A"
                        st.metric("🏢 Market Cap", cap_display)
                    
                    with col4:
                        volume = stock_data['volume']
                        if volume >= 1e6:
                            volume_display = f"{volume/1e6:.2f}M"
                        elif volume >= 1e3:
                            volume_display = f"{volume/1e3:.2f}K"
                        else:
                            volume_display = str(volume)
                        st.metric("📊 Volume", volume_display)
                    
                    # Company Info
                    company_info = stock_data['company_info']
                    if company_info.get('longName'):
                        st.subheader(f"📋 {company_info['longName']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if company_info.get('sector'):
                                st.write(f"**Sector:** {company_info['sector']}")
                            if company_info.get('industry'):
                                st.write(f"**Industry:** {company_info['industry']}")
                        
                        with col2:
                            if company_info.get('country'):
                                st.write(f"**Country:** {company_info['country']}")
                            if company_info.get('employees'):
                                st.write(f"**Employees:** {company_info['employees']:,}")
                    
                    # Technical Analysis
                    if show_technical:
                        st.subheader("🔧 Technical Analysis")
                        
                        technical_data = analyzer.calculate_technical_indicators(stock_data['price_data'])
                        
                        if technical_data:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**Moving Averages**")
                                if technical_data.get('sma_20'):
                                    st.write(f"SMA 20: ${technical_data['sma_20']:.2f}")
                                if technical_data.get('sma_50'):
                                    st.write(f"SMA 50: ${technical_data['sma_50']:.2f}")
                                if technical_data.get('sma_200'):
                                    st.write(f"SMA 200: ${technical_data['sma_200']:.2f}")
                            
                            with col2:
                                st.write("**Momentum Indicators**")
                                rsi = technical_data.get('rsi')
                                if rsi:
                                    rsi_status = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "🟡 Neutral"
                                    st.write(f"RSI: {rsi:.1f} {rsi_status}")
                                
                                macd = technical_data.get('macd')
                                macd_signal = technical_data.get('macd_signal')
                                if macd and macd_signal:
                                    macd_status = "🟢 Bullish" if macd > macd_signal else "🔴 Bearish"
                                    st.write(f"MACD: {macd_status}")
                            
                            with col3:
                                st.write("**Support & Resistance**")
                                if technical_data.get('bb_upper') and technical_data.get('bb_lower'):
                                    st.write(f"BB Upper: ${technical_data['bb_upper']:.2f}")
                                    st.write(f"BB Lower: ${technical_data['bb_lower']:.2f}")
                    
                    # Fundamental Analysis
                    if show_fundamental:
                        st.subheader("💼 Fundamental Analysis")
                        
                        fundamental_data = analyzer.calculate_fundamental_metrics(stock_data['company_info'])
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Valuation Metrics**")
                            pe = fundamental_data.get('pe_ratio')
                            st.write(f"P/E Ratio: {pe if pe == 'N/A' else f'{pe:.2f}'}")
                            
                            pb = fundamental_data.get('price_to_book')
                            st.write(f"P/B Ratio: {pb if pb == 'N/A' else f'{pb:.2f}'}")
                            
                            ps = fundamental_data.get('price_to_sales')
                            st.write(f"P/S Ratio: {ps if ps == 'N/A' else f'{ps:.2f}'}")
                        
                        with col2:
                            st.write("**Financial Health**")
                            roe = fundamental_data.get('return_on_equity')
                            if isinstance(roe, (int, float)):
                                st.write(f"ROE: {roe*100:.1f}%")
                            else:
                                st.write(f"ROE: {roe}")
                            
                            debt = fundamental_data.get('debt_to_equity')
                            st.write(f"Debt/Equity: {debt if debt == 'N/A' else f'{debt:.2f}'}")
                            
                            beta = fundamental_data.get('beta')
                            st.write(f"Beta: {beta if beta == 'N/A' else f'{beta:.2f}'}")
                        
                        with col3:
                            st.write("**Growth Metrics**")
                            rev_growth = fundamental_data.get('revenue_growth')
                            if isinstance(rev_growth, (int, float)):
                                st.write(f"Revenue Growth: {rev_growth*100:.1f}%")
                            else:
                                st.write(f"Revenue Growth: {rev_growth}")
                            
                            div_yield = fundamental_data.get('dividend_yield')
                            if isinstance(div_yield, (int, float)):
                                st.write(f"Dividend Yield: {div_yield*100:.1f}%")
                            else:
                                st.write(f"Dividend Yield: {div_yield}")
                    
                    # AI Recommendation
                    if show_ai_recommendation:
                        st.subheader("🤖 AI-Powered Recommendation")
                        
                        technical_data = analyzer.calculate_technical_indicators(stock_data['price_data'])
                        fundamental_data = analyzer.calculate_fundamental_metrics(stock_data['company_info'])
                        
                        ai_analysis = analyzer.generate_ai_recommendation(
                            symbol_input, technical_data, fundamental_data, stock_data['price_data']
                        )
                        
                        # Recommendation Display
                        recommendation = ai_analysis['recommendation']
                        
                        if recommendation in ['STRONG BUY', 'BUY']:
                            rec_class = "recommendation-buy"
                        elif recommendation == 'HOLD':
                            rec_class = "recommendation-hold"
                        else:
                            rec_class = "recommendation-sell"
                        
                        st.markdown(f'''
                        <div class="{rec_class}">
                            <h3>🎯 {recommendation}</h3>
                            <p><strong>Confidence:</strong> {ai_analysis['confidence']}</p>
                            <p><strong>Overall Score:</strong> {ai_analysis['overall_score']}/1.0</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Score Breakdown and Price Targets
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📊 Score Breakdown**")
                            st.write(f"Technical Score: {ai_analysis['technical_score']}")
                            st.write(f"Fundamental Score: {ai_analysis['fundamental_score']}")
                            st.write(f"Overall Score: {ai_analysis['overall_score']}")
                        
                        with col2:
                            st.write("**🎯 Price Targets**")
                            targets = ai_analysis['price_targets']
                            st.write(f"Current Price: ${targets['current']}")
                            st.write(f"Target High: ${targets['target_high']}")
                            st.write(f"Target Low: ${targets['target_low']}")
                            st.write(f"Stop Loss: ${targets['stop_loss']}")
                        
                        # Key Insights
                        if ai_analysis['key_insights']:
                            st.write("**💡 Key Insights**")
                            for insight in ai_analysis['key_insights']:
                                st.write(f"• {insight}")
                    
                    # Price Chart
                    st.subheader("📈 Price Chart & Technical Indicators")
                    
                    fig = go.Figure()
                    
                    # Candlestick Chart
                    fig.add_trace(go.Candlestick(
                        x=stock_data['price_data'].index,
                        open=stock_data['price_data']['Open'],
                        high=stock_data['price_data']['High'],
                        low=stock_data['price_data']['Low'],
                        close=stock_data['price_data']['Close'],
                        name=symbol_input,
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    ))
                    
                    # Add Moving Averages
                    if show_technical and technical_data:
                        if technical_data.get('sma_20'):
                            sma_20_series = stock_data['price_data']['Close'].rolling(20).mean()
                            fig.add_trace(go.Scatter(
                                x=stock_data['price_data'].index,
                                y=sma_20_series,
                                mode='lines',
                                name='SMA 20',
                                line=dict(color='orange', width=1)
                            ))
                        
                        if technical_data.get('sma_50'):
                            sma_50_series = stock_data['price_data']['Close'].rolling(50).mean()
                            fig.add_trace(go.Scatter(
                                x=stock_data['price_data'].index,
                                y=sma_50_series,
                                mode='lines',
                                name='SMA 50',
                                line=dict(color='blue', width=1)
                            ))
                        
                        # Bollinger Bands
                        if technical_data.get('bb_upper') and technical_data.get('bb_lower'):
                            bb_upper_series = stock_data['price_data']['Close'].rolling(20).mean() + (stock_data['price_data']['Close'].rolling(20).std() * 2)
                            bb_lower_series = stock_data['price_data']['Close'].rolling(20).mean() - (stock_data['price_data']['Close'].rolling(20).std() * 2)
                            
                            fig.add_trace(go.Scatter(
                                x=stock_data['price_data'].index,
                                y=bb_upper_series,
                                mode='lines',
                                name='BB Upper',
                                line=dict(color='gray', width=1, dash='dash')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=stock_data['price_data'].index,
                                y=bb_lower_series,
                                mode='lines',
                                name='BB Lower',
                                line=dict(color='gray', width=1, dash='dash'),
                                fill='tonexty',
                                fillcolor='rgba(128,128,128,0.1)'
                            ))
                    
                    fig.update_layout(
                        title=f'{symbol_input} - Price Chart with Technical Indicators',
                        yaxis_title='Price ($)',
                        xaxis_title='Date',
                        height=600,
                        xaxis_rangeslider_visible=False,
                        showlegend=True
                    )
                    
                    st.plotly_chart(volume_fig, use_container_width=True)
                
                else:
                    st.error(f"❌ Could not fetch data for {symbol_input}. Please check the symbol and try again.")
        
        elif symbol_input:
            st.info("👆 Click the 'Analyze Stock' button to start the analysis!")
    
    # Tab 2: Portfolio Management
    with tab2:
        st.header("💼 Portfolio Management")
        
        # Add Holdings Section
        st.subheader("➕ Add New Holding")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            portfolio_symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL").upper()
        
        with col2:
            shares = st.number_input("Number of Shares", min_value=0.0, step=0.1, format="%.2f")
        
        with col3:
            cost_basis = st.number_input("Cost Basis per Share ($)", min_value=0.0, step=0.01, format="%.2f")
        
        with col4:
            st.write("")  # Spacing
            if st.button("📈 Add to Portfolio", type="primary"):
                if portfolio_symbol and shares > 0 and cost_basis > 0:
                    if 'portfolio_holdings' not in st.session_state:
                        st.session_state.portfolio_holdings = {}
                    
                    if portfolio_symbol in st.session_state.portfolio_holdings:
                        # Update existing holding
                        existing = st.session_state.portfolio_holdings[portfolio_symbol]
                        total_shares = existing['shares'] + shares
                        total_cost = (existing['shares'] * existing['cost_basis']) + (shares * cost_basis)
                        new_avg_cost = total_cost / total_shares
                        
                        st.session_state.portfolio_holdings[portfolio_symbol] = {
                            'shares': total_shares,
                            'cost_basis': new_avg_cost
                        }
                        st.success(f"✅ Updated {portfolio_symbol}: {total_shares} shares at avg cost ${new_avg_cost:.2f}")
                    else:
                        # Add new holding
                        st.session_state.portfolio_holdings[portfolio_symbol] = {
                            'shares': shares,
                            'cost_basis': cost_basis
                        }
                        st.success(f"✅ Added {shares} shares of {portfolio_symbol} at ${cost_basis:.2f}")
                else:
                    st.error("❌ Please fill in all fields with valid values.")
        
        # Display Portfolio
        if 'portfolio_holdings' in st.session_state and st.session_state.portfolio_holdings:
            st.subheader("📊 Current Portfolio")
            
            # Get current prices for all holdings
            current_prices = {}
            for symbol in st.session_state.portfolio_holdings.keys():
                stock_data = analyzer.get_stock_data(symbol, period="1d")
                if stock_data:
                    current_prices[symbol] = stock_data['current_price']
            
            # Calculate portfolio metrics
            portfolio_metrics = portfolio_manager.calculate_portfolio_metrics(
                st.session_state.portfolio_holdings, current_prices
            )
            
            if portfolio_metrics:
                # Portfolio Summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "💰 Total Value", 
                        f"${portfolio_metrics['total_value']:,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "💸 Total Cost", 
                        f"${portfolio_metrics['total_cost']:,.2f}"
                    )
                
                with col3:
                    total_return = portfolio_metrics['total_return']
                    st.metric(
                        "📈 Total Return", 
                        f"{total_return:.2f}%",
                        delta=f"{total_return:.2f}%"
                    )
                
                with col4:
                    gain_loss = portfolio_metrics['total_gain_loss']
                    st.metric(
                        "💵 Gain/Loss", 
                        f"${gain_loss:,.2f}",
                        delta=f"${gain_loss:,.2f}"
                    )
                
                # Portfolio Holdings Table
                st.subheader("📋 Holdings Details")
                
                portfolio_data = []
                for position in portfolio_metrics['positions']:
                    portfolio_data.append({
                        'Symbol': position['symbol'],
                        'Shares': f"{position['shares']:.2f}",
                        'Cost Basis': f"${position['cost_basis']:.2f}",
                        'Current Price': f"${position['current_price']:.2f}",
                        'Position Value': f"${position['position_value']:,.2f}",
                        'Gain/Loss': f"${position['gain_loss']:,.2f}",
                        'Return %': f"{position['gain_loss_pct']:.2f}%"
                    })
                
                df = pd.DataFrame(portfolio_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Portfolio Allocation Chart
                if len(portfolio_data) > 1:
                    st.subheader("🥧 Portfolio Allocation")
                    
                    allocation_data = []
                    for position in portfolio_metrics['positions']:
                        allocation_data.append({
                            'Symbol': position['symbol'],
                            'Value': position['position_value'],
                            'Percentage': (position['position_value'] / portfolio_metrics['total_value']) * 100
                        })
                    
                    fig_pie = px.pie(
                        pd.DataFrame(allocation_data),
                        values='Value',
                        names='Symbol',
                        title='Portfolio Allocation by Value',
                        hover_data=['Percentage']
                    )
                    
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label'
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Portfolio Actions
                st.subheader("⚙️ Portfolio Actions")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Rebalance Portfolio"):
                        st.info("💡 Rebalancing feature coming soon!")
                
                with col2:
                    if st.button("📤 Export Portfolio"):
                        portfolio_json = json.dumps(st.session_state.portfolio_holdings, indent=2)
                        st.download_button(
                            label="💾 Download JSON",
                            data=portfolio_json,
                            file_name=f"papaistra_portfolio_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json"
                        )
                
                with col3:
                    if st.button("🗑️ Clear Portfolio"):
                        if st.button("⚠️ Confirm Clear", type="secondary"):
                            st.session_state.portfolio_holdings = {}
                            st.success("✅ Portfolio cleared!")
                            st.rerun()
            else:
                st.warning("⚠️ Unable to calculate portfolio metrics. Please check your holdings.")
        
        else:
            st.info("📝 Add some holdings above to start tracking your portfolio performance!")
    
    # Tab 3: Market Overview
    with tab3:
        st.header("🌍 Global Market Overview")
        
        # Major Market Indices
        st.subheader("📊 Major Market Indices")
        
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000'
        }
        
        cols = st.columns(len(indices))
        
        for i, (symbol, name) in enumerate(indices.items()):
            with cols[i]:
                try:
                    index_data = analyzer.get_stock_data(symbol, period="5d")
                    if index_data:
                        st.metric(
                            name,
                            f"{index_data['current_price']:.2f}",
                            delta=f"{index_data['daily_change']:.2f}%"
                        )
                    else:
                        st.metric(name, "Loading...")
                except:
                    st.metric(name, "Error")
        
        # Market Sectors Performance
        st.subheader("🏭 Sector Performance")
        
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financial',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrial',
            'XLY': 'Consumer Disc.',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities'
        }
        
        sector_data = []
        
        for etf, sector in sector_etfs.items():
            try:
                data = analyzer.get_stock_data(etf, period="1d")
                if data:
                    sector_data.append({
                        'Sector': sector,
                        'ETF': etf,
                        'Price': f"${data['current_price']:.2f}",
                        'Change %': f"{data['daily_change']:.2f}%",
                        'Change': data['daily_change']
                    })
            except:
                continue
        
        if sector_data:
            # Sort by performance
            sector_df = pd.DataFrame(sector_data)
            sector_df = sector_df.sort_values('Change', ascending=False)
            
            # Display sector performance
            st.dataframe(
                sector_df[['Sector', 'ETF', 'Price', 'Change %']], 
                use_container_width=True, 
                hide_index=True
            )
            
            # Sector performance chart
            fig_sectors = px.bar(
                sector_df,
                x='Sector',
                y='Change',
                title='Sector Performance Today (%)',
                color='Change',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            
            fig_sectors.update_layout(height=400)
            st.plotly_chart(fig_sectors, use_container_width=True)
        
        # Market Highlights
        st.subheader("🔥 Market Highlights")
        
        # Sample market movers (in a real app, this would be live data)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🚀 Top Gainers**")
            gainers_data = [
                {'Symbol': 'NVDA', 'Price': '$875.43', 'Change': '+5.2%'},
                {'Symbol': 'TSLA', 'Price': '$248.92', 'Change': '+4.8%'},
                {'Symbol': 'AMZN', 'Price': '$182.15', 'Change': '+3.1%'},
                {'Symbol': 'MSFT', 'Price': '$415.26', 'Change': '+2.9%'},
                {'Symbol': 'GOOGL', 'Price': '$175.84', 'Change': '+2.3%'}
            ]
            st.dataframe(pd.DataFrame(gainers_data), hide_index=True, use_container_width=True)
        
        with col2:
            st.write("**📉 Top Losers**")
            losers_data = [
                {'Symbol': 'META', 'Price': '$494.32', 'Change': '-2.8%'},
                {'Symbol': 'NFLX', 'Price': '$641.28', 'Change': '-2.1%'},
                {'Symbol': 'AAPL', 'Price': '$192.53', 'Change': '-1.9%'},
                {'Symbol': 'CRM', 'Price': '$284.67', 'Change': '-1.5%'},
                {'Symbol': 'AMD', 'Price': '$144.29', 'Change': '-1.2%'}
            ]
            st.dataframe(pd.DataFrame(losers_data), hide_index=True, use_container_width=True)
        
        # Economic Calendar
        st.subheader("📅 Economic Calendar")
        st.info("💡 Economic calendar integration coming soon! This will show upcoming earnings, Fed meetings, and economic indicators.")
    
    # Tab 4: Performance Analytics
    with tab4:
        st.header("📈 Performance Analytics")
        
        # Portfolio Performance Over Time
        if 'portfolio_holdings' in st.session_state and st.session_state.portfolio_holdings:
            st.subheader("📊 Portfolio Performance Tracking")
            
            # Simulated performance data (in a real app, this would be historical data)
            dates = pd.date_range(start='2024-01-01', periods=180, freq='D')
            
            # Generate realistic portfolio performance
            np.random.seed(42)  # For consistent demo data
            daily_returns = np.random.normal(0.0008, 0.02, 180)  # ~20% annual return, 20% volatility
            portfolio_values = 100000 * np.cumprod(1 + daily_returns)
            
            # Generate benchmark performance (S&P 500)
            benchmark_returns = np.random.normal(0.0006, 0.015, 180)  # ~15% annual return, 15% volatility
            benchmark_values = 100000 * np.cumprod(1 + benchmark_returns)
            
            performance_df = pd.DataFrame({
                'Date': dates,
                'Portfolio': portfolio_values,
                'S&P 500': benchmark_values
            })
            
            # Performance Chart
            fig_performance = go.Figure()
            
            fig_performance.add_trace(go.Scatter(
                x=performance_df['Date'],
                y=performance_df['Portfolio'],
                mode='lines',
                name='Your Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            fig_performance.add_trace(go.Scatter(
                x=performance_df['Date'],
                y=performance_df['S&P 500'],
                mode='lines',
                name='S&P 500 Benchmark',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_performance.update_layout(
                title='Portfolio Performance vs Benchmark',
                yaxis_title='Portfolio Value ($)',
                xaxis_title='Date',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Performance Metrics
            st.subheader("📊 Performance Metrics")
            
            # Calculate metrics
            portfolio_return = ((portfolio_values[-1] / portfolio_values[0]) - 1) * 100
            benchmark_return = ((benchmark_values[-1] / benchmark_values[0]) - 1) * 100
            alpha = portfolio_return - benchmark_return
            
            portfolio_vol = np.std(daily_returns) * np.sqrt(252) * 100
            benchmark_vol = np.std(benchmark_returns) * np.sqrt(252) * 100
            
            sharpe_ratio = (np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📈 Total Return",
                    f"{portfolio_return:.2f}%",
                    delta=f"{alpha:.2f}% vs S&P 500"
                )
            
            with col2:
                st.metric(
                    "📊 Volatility",
                    f"{portfolio_vol:.2f}%"
                )
            
            with col3:
                st.metric(
                    "⚡ Sharpe Ratio",
                    f"{sharpe_ratio:.2f}"
                )
            
            with col4:
                max_drawdown = ((portfolio_values / np.maximum.accumulate(portfolio_values)) - 1).min() * 100
                st.metric(
                    "📉 Max Drawdown",
                    f"{max_drawdown:.2f}%"
                )
            
            # Risk-Return Scatter Plot
            st.subheader("🎯 Risk-Return Analysis")
            
            # Calculate rolling 30-day returns and volatilities for scatter plot
            rolling_window = 30
            rolling_returns = []
            rolling_vols = []
            
            for i in range(rolling_window, len(daily_returns)):
                window_returns = daily_returns[i-rolling_window:i]
                rolling_returns.append(np.mean(window_returns) * 252 * 100)
                rolling_vols.append(np.std(window_returns) * np.sqrt(252) * 100)
            
            risk_return_df = pd.DataFrame({
                'Return (%)': rolling_returns,
                'Volatility (%)': rolling_vols,
                'Date': dates[rolling_window:]
            })
            
            fig_scatter = px.scatter(
                risk_return_df,
                x='Volatility (%)',
                y='Return (%)',
                title='Risk-Return Profile (30-day rolling)',
                hover_data=['Date']
            )
            
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        else:
            st.info("📝 Add holdings to your portfolio to see performance analytics!")
        
        # Analysis History
        st.subheader("📋 Recent Analysis History")
        
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        if st.session_state.analysis_history:
            history_df = pd.DataFrame(st.session_state.analysis_history)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("🔍 Start analyzing stocks to build your analysis history!")

if __name__ == "__main__":
    # Initialize session state
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = ''
    
    if 'portfolio_holdings' not in st.session_state:
        st.session_state.portfolio_holdings = {}
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Run the main application
    main()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; padding: 1rem;">'
        '🚀 <strong>PapAIstra v1.0</strong> | AI-Powered Investment Platform | '
        'Built with ❤️ using Streamlit'
        '</div>',
        unsafe_allow_html=True
    )fig, use_container_width=True)
                    
                    # Volume Chart
                    volume_fig = go.Figure()
                    volume_fig.add_trace(go.Bar(
                        x=stock_data['price_data'].index,
                        y=stock_data['price_data']['Volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ))
                    
                    volume_fig.update_layout(
                        title=f'{symbol_input} - Trading Volume',
                        yaxis_title='Volume',
                        xaxis_title='Date',
                        height=300
                    )
                    
                    st.plotly_chart(