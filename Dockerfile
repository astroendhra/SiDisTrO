FROM pytorch/pytorch:latest

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "ddp_launch.py"]