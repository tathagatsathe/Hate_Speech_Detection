FROM python:3.9-slim-buster
WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run" ]
CMD ["streamlit_app.py"]