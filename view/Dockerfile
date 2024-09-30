# Use a lightweight Python image with version 3.9
FROM python:3.9-slim

# Create a non-root user with UID 1000 as required by Hugging Face Spaces
RUN useradd -m -u 1000 user

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/user/.local/bin:$PATH"

# Install required system packages (including wget)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git
RUN git config --global user.email "Mojmir@users.noreply.huggingface.co"
# Switch to non-root user
USER user

# Set the working directory (root, since you don’t have an /app folder)
WORKDIR /home/user

# Copy requirements.txt before copying other files to utilize Docker cache
COPY --chown=user requirements.txt .

# pip update install tools
run pip install -U pip setuptools wheel
# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files to the container
COPY --chown=user . .

# RUN wget https://github.com/gitpod-io/openvscode-server/releases/download/openvscode-server-v1.86.2/openvscode-server-v1.86.2-linux-x64.tar.gz -O /tmp/openvscode-server.tar.gz && \
#     tar -xzf /tmp/openvscode-server.tar.gz -C /opt && \
#     rm /tmp/openvscode-server.tar.gz && \
#     mv /opt/openvscode-server-v1.86.2-linux-x64 /opt/openvscode-server && \
#     chown -R 1000:1000 /opt/openvscode-server

# Expose the port Gradio runs on (7860)
# EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "app.py"]