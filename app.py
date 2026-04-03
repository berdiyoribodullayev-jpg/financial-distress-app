import streamlit as st

st.title("Financial Distress App")

st.write("Enter a company ticker to begin analysis.")

ticker = st.text_input("Company ticker", placeholder="e.g. AAPL")

if st.button("Analyze"):
    if ticker:
        st.success(f"You entered: {ticker.upper()}")
    else:
        st.warning("Please enter a ticker symbol.")