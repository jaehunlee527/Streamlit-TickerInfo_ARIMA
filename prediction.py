from functions import *

def prediction():
    st.markdown("# ARIMA Forecasting")
    company = st.sidebar.text_input("Company Ticker")

    st.text("")
    st.text("")
    start = st.sidebar.date_input("Start Date")
    end = st.sidebar.date_input("End Date")
    day_range = st.sidebar.number_input("Prediction Day Range", 0, 200, value=0, step=10) 

    update = st.sidebar.button("Update")

    if update:
    	with st.spinner("Fetching Company Profile..."):
    		df, df_close = df_caller(company)
    	with st.spinner("Prediction in Progress..."):
            df = df_slicer(df, start, end)
            df_close = df_slicer(df_close, start, end)
            best_param = pdq_calc(df_close)

    	predict(df_close, best_param, day_range)




