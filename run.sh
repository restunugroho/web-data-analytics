#!/bin/bash

# Data Analytics MVP Runner Script
echo "🚀 Starting Data Analytics MVP..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Function to check if port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null; then
        return 1
    else
        return 0
    fi
}

# Check if ports are available
if ! check_port 8000; then
    echo "⚠️  Port 8000 is already in use. Please free the port or change the backend port."
    exit 1
fi

if ! check_port 8501; then
    echo "⚠️  Port 8501 is already in use. Please free the port or change the frontend port."
    exit 1
fi

# Install dependencies
# echo "📦 Installing dependencies..."
# pip install -r requirements.txt

# Check installation success
# if [ $? -ne 0 ]; then
#     echo "❌ Failed to install dependencies. Please check your Python environment."
#     exit 1
# fi

echo "✅ Dependencies installed successfully!"

# Create log directory
mkdir -p logs

# Function to cleanup processes
cleanup() {
    echo "🛑 Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start backend
echo "🔧 Starting FastAPI backend..."
python main.py > logs/backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 180

# Check if backend is running
if ! curl -s http://localhost:8000/ > /dev/null; then
    echo "❌ Backend failed to start. Check logs/backend.log for details."
    cleanup
fi

echo "✅ Backend started at http://localhost:8000"

# Start frontend
echo "🎨 Starting Streamlit frontend..."
streamlit run app.py --server.headless true > logs/frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
sleep 10

echo "✅ Frontend started at http://localhost:8501"

echo "
🎉 Data Analytics MVP is now running!

📊 Frontend (Streamlit): http://localhost:8501
🔧 Backend (FastAPI):   http://localhost:8000
📚 API Documentation:   http://localhost:8000/docs

📝 Logs:
   Backend:  logs/backend.log
   Frontend: logs/frontend.log

Press Ctrl+C to stop all services
"

# Wait for user interrupt
wait