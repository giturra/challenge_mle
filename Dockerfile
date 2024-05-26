FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the content of the current directory to the working directory in the container
COPY . .

# Expose port 8080 to access the application
EXPOSE 8080

# Command to run the FastAPI application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]