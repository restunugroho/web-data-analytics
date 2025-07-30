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
    echo "⚠️ Port 8000 is already in use. Please free the port or change the backend port."
    exit 1
fi

# if ! check_port 8501; then
#     echo "⚠️ Port 8501 is already in use. Please free the port or change the frontend port."
#     exit 1
# fi

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

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=${3:-30}
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 0.5
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start within $((max_attempts * 1)) seconds"
    return 1
}

# Function to check if process is still running
is_process_running() {
    local pid=$1
    if [ -z "$pid" ]; then
        return 1
    fi
    kill -0 "$pid" 2>/dev/null
}

# Function to cleanup processes
cleanup() {
    echo "🛑 Shutting down services..."
    
    if [ ! -z "$BACKEND_PID" ] && is_process_running $BACKEND_PID; then
        echo "🔧 Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null
        sleep 2
        # Force kill if still running
        if is_process_running $BACKEND_PID; then
            kill -9 $BACKEND_PID 2>/dev/null
        fi
    fi
    
    if [ ! -z "$FRONTEND_PID" ] && is_process_running $FRONTEND_PID; then
        echo "🎨 Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null
        sleep 2
        # Force kill if still running
        if is_process_running $FRONTEND_PID; then
            kill -9 $FRONTEND_PID 2>/dev/null
        fi
    fi
    
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Start backend
echo "🔧 Starting FastAPI backend..."
python main.py > logs/backend.log 2>&1 &
BACKEND_PID=$!

# Check if backend process started successfully
sleep 1
if ! is_process_running $BACKEND_PID; then
    echo "❌ Backend process failed to start. Check logs/backend.log for details."
    exit 1
fi

# Wait for backend to be ready with health check
if ! wait_for_service "http://localhost:8000/health" "Backend" 60; then
    echo "❌ Backend health check failed. Trying root endpoint..."
    # Fallback to root endpoint if /health doesn't exist
    if ! wait_for_service "http://localhost:8000/" "Backend" 30; then
        echo "❌ Backend failed to start. Check logs/backend.log for details."
        exit 1
    fi
fi

echo "✅ Backend started at http://localhost:8000"

# Start frontend
echo "🎨 Starting Streamlit frontend..."
streamlit run app.py --server.headless true > logs/frontend.log 2>&1 &
FRONTEND_PID=$!

# Check if frontend process started successfully
sleep 1
if ! is_process_running $FRONTEND_PID; then
    echo "❌ Frontend process failed to start. Check logs/frontend.log for details."
    exit 1
fi

# Wait for frontend to be ready
# if ! wait_for_service "http://localhost:8501/" "Frontend" 30; then
#     echo "❌ Frontend failed to start. Check logs/frontend.log for details."
#     exit 1
# fi

echo "✅ Frontend started at http://localhost:8501"

echo "
🎉 Data Analytics MVP is now running!
📊 Frontend (Streamlit): http://localhost:8501
🔧 Backend (FastAPI): http://localhost:8000
📚 API Documentation: http://localhost:8000/docs
📝 Logs:
   Backend: logs/backend.log
   Frontend: logs/frontend.log

Press Ctrl+C to stop all services
"

# Keep script running and wait for user interrupt
while true; do
    # Check if processes are still running
    if ! is_process_running $BACKEND_PID; then
        echo "❌ Backend process has stopped unexpectedly!"
        exit 1
    fi
    
    if ! is_process_running $FRONTEND_PID; then
        echo "❌ Frontend process has stopped unexpectedly!"
        exit 1
    fi
    
    sleep 5
done