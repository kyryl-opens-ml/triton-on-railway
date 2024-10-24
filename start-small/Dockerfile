FROM python:3.12

WORKDIR /app
RUN pip install nvidia-pytriton==0.5.12

ENV PYTHONPATH /app
COPY server.py server.py 

CMD python /app/server.py
