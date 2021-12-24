import show_price
import prediction
import streamlit as st
st.set_page_config(layout="wide")

def main():
	options = ["Historical Price", "ARIMA Forecast"]
	view = st.sidebar.radio("View", options)
	
	if view == "Historical Price":
		show_price.show_price()	
	if view == "ARIMA Forecast":
		prediction.prediction()

if __name__ =="__main__":
	main()
