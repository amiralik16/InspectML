FROM python:3.7

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./api /api
COPY ./model /model

EXPOSE 80

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]