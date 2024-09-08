# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 29500 available to the world outside this container
EXPOSE 29500

# Run ddp_launch.py when the container launches
CMD ["python", "ddp_launch.py"]
