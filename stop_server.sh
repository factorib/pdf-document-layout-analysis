#!/bin/bash

# A10G GPU-Optimized PDF Layout Analysis Server Stop Script

echo "🛑 Stopping A10G GPU-Optimized PDF Layout Analysis Server"
echo "==========================================================="

# Stop server using PID file
if [ -f "pids/server.pid" ]; then
    SERVER_PID=$(cat pids/server.pid 2>/dev/null || echo "")
    if [ ! -z "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "📍 Found server process (PID: $SERVER_PID)"
        echo "⏳ Gracefully stopping server..."
        kill "$SERVER_PID"
        sleep 3
        
        # Check if still running
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "⚡ Force stopping server..."
            kill -9 "$SERVER_PID"
            sleep 1
        fi
        
        # Verify stopped
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "✅ Server stopped successfully"
            rm -f pids/server.pid
        else
            echo "❌ Failed to stop server"
            exit 1
        fi
    else
        echo "⚠️  PID file exists but process not found"
        rm -f pids/server.pid
    fi
else
    echo "⚠️  No PID file found"
fi

# Kill any remaining process on port 80
echo "🔍 Checking for any remaining processes on port 80..."
if sudo fuser -k 80/tcp 2>/dev/null; then
    echo "✅ Stopped additional processes on port 80"
else
    echo "ℹ️  No additional processes found on port 80"
fi

echo "🏁 Server shutdown complete"