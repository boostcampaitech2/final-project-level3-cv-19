run_black:
	python3 -m black . -l 119

run_server:
	python3 -m app

run_client:
	python3 -m streamlit run app/frontend.py --server.port 6006
 
run_app: run_server run_client
