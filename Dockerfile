FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ports
EXPOSE 8501 8000

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting NBA Predictor Production System..."\n\
echo "Streamlit Demo: http://localhost:8501"\n\
echo "Flask API: http://localhost:8000"\n\
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 &\n\
sleep 5\n\
echo "Production system started successfully!"\n\
wait' > /app/start.sh

RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start the application
CMD ["/app/start.sh"]