from functions import *

def show_price():
	st.text("")
	
	company = st.sidebar.text_input("Company Ticker")
	start_date = st.sidebar.date_input("Start Date")
	end_date = st.sidebar.date_input("End Date")
	show_ma = st.sidebar.checkbox("Moving Average")
	
	days = 0
	
	if show_ma:
		days = st.sidebar.number_input("Days", value=0, min_value=0, max_value=200, step=1)
	
	show_pct = st.sidebar.checkbox("Daily Percentage Change")
	show_macd = st.sidebar.checkbox("MACD")
	show_bollinger = st.sidebar.checkbox("Bollinger Band")
	show_rsi = st.sidebar.checkbox("RSI")

	update = st.sidebar.button("Update")

	if end_date < start_date:
		st.error("Invalid date")
	
	if update:
		df, df_close = df_caller(company)
		df = df_slicer(df, start_date, end_date)
		st.markdown("# Ticker Info")
		st.markdown("")

		if len(df) > 0:
			st.text("")
			with st.spinner("Loading..."):
				plot_price(df, company, days, show_bollinger)
		
		if show_pct:
			st.text("")
			with st.spinner("Loading..."):
				plot_pct(df)

		if show_macd or show_rsi:
			st.text("")
			with st.spinner("Loading..."):
				plot_ind(df, show_macd, show_rsi)



