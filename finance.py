import streamlit as st
import yfinance as yf
import pandas as pd

# Function to fetch financial data
def get_financials(ticker):
    stock = yf.Ticker(ticker)
    income_statement = stock.financials.T
    balance_sheet = stock.balance_sheet.T
    cash_flow = stock.cashflow.T
    return income_statement, balance_sheet, cash_flow

# Function to calculate financial ratios
def calculate_ratios(balance_sheet, income_statement, cash_flow):
    # Debugging step: Display column names
    st.write("Balance Sheet Columns:", balance_sheet.columns.tolist())
    st.write("Income Statement Columns:", income_statement.columns.tolist())
    st.write("Cash Flow Statement Columns:", cash_flow.columns.tolist())
    
    # Liquidity Ratios
    current_ratio = balance_sheet['Current Assets'] / balance_sheet['Current Liabilities']
    quick_ratio = (balance_sheet['Current Assets'] - balance_sheet['Inventory']) / balance_sheet['Current Liabilities']
    
    # Solvency Ratios
    debt_to_equity = balance_sheet['Total Liabilities Net Minority Interest'] / balance_sheet['Stockholders Equity']
    interest_coverage = income_statement['EBIT'] / income_statement['Interest Expense']

    # Add calculated ratios to a DataFrame
    ratios = pd.DataFrame({
        'Current Ratio': current_ratio,
        'Quick Ratio': quick_ratio,
        'Debt to Equity': debt_to_equity,
        'Interest Coverage': interest_coverage
    })

    return ratios

# Streamlit App
st.title("Financial Statement Analysis")
ticker = st.text_input("Enter stock ticker:", "AAPL")

if ticker:
    income_statement, balance_sheet, cash_flow = get_financials(ticker)
    
    # Display Income Statement
    st.header("Income Statement")
    st.dataframe(income_statement)
    
    # Display Balance Sheet
    st.header("Balance Sheet")
    st.dataframe(balance_sheet)
    
    # Display Cash Flow Statement
    st.header("Cash Flow Statement")
    st.dataframe(cash_flow)
    
    # Calculate and display financial ratios
    st.header("Financial Ratios")
    ratios = calculate_ratios(balance_sheet, income_statement, cash_flow)
    st.dataframe(ratios)

    # Additional specific metrics
    st.header("Specific Metrics")
    revenue = income_statement['Total Revenue']
    cogs = income_statement['Cost Of Revenue']
    gross_profit_margin = (income_statement['Gross Profit'] / income_statement['Total Revenue']) * 100
    operating_expenses = income_statement[['Selling General And Administration', 'Research And Development']].sum(axis=1)
    operating_income = income_statement['Operating Income']
    net_income = income_statement['Net Income']
    
    specific_metrics = pd.DataFrame({
        'Revenue': revenue,
        'COGS': cogs,
        'Gross Profit Margin (%)': gross_profit_margin,
        'Operating Expenses': operating_expenses,
        'Operating Income': operating_income,
        'Net Income': net_income
    })
    st.dataframe(specific_metrics)

    # Free Cash Flow calculation
    st.header("Free Cash Flow")
    st.write("Cash Flow Statement Columns:", cash_flow.columns.tolist())  # Display the columns for inspection
    try:
        # Identify the correct column names for cash flow calculations
        operating_cash_flow = cash_flow['Total Cash From Operating Activities'] if 'Total Cash From Operating Activities' in cash_flow.columns else cash_flow['Operating Cash Flow']
        capital_expenditures = cash_flow['Capital Expenditures'] if 'Capital Expenditures' in cash_flow.columns else cash_flow['Capital Expenditure']
        
        free_cash_flow = operating_cash_flow - capital_expenditures
        st.dataframe(free_cash_flow)
    except KeyError as e:
        st.error(f"Key error: {e}. Please check the column names in the Cash Flow Statement.")
