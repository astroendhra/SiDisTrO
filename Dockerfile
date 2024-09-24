FROM pytorch/pytorch:latest

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Use an entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]