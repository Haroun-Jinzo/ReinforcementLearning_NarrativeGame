"""
Detective Game - Python Server (Week 1 Starter)
Simple Flask server to test Unity <-> Python communication

Run with: python server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for Unity

print("=" * 50)
print("🕵️ DETECTIVE GAME - PYTHON SERVER")
print("=" * 50)

# ==================== TEST ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint - test if server is running"""
    return jsonify({
        'status': 'running',
        'message': 'Python server is alive! 🎉',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/test', methods=['POST'])
def test():
    """Test endpoint - echoes back Unity's message"""
    try:
        # Get data from Unity
        data = request.json
        print(f"\n📨 Received from Unity: {data}")
        
        # Extract message
        unity_message = data.get('message', 'No message sent')
        
        # Create response
        response = {
            'status': 'success',
            'message': f"Hello Unity! You sent: '{unity_message}'",
            'timestamp': time.time(),
            'python_version': '3.x',
            'server_status': 'operational'
        }
        
        print(f"📤 Sending response: {response['message']}")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error in /test endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ==================== FUTURE ENDPOINTS (Week 2+) ====================

@app.route('/start_game', methods=['POST'])
def start_game():
    """Initialize a new game session (Week 2)"""
    return jsonify({
        'status': 'not_implemented',
        'message': 'This endpoint will be implemented in Week 2'
    })


@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Handle detective questions (Week 2)"""
    return jsonify({
        'status': 'not_implemented',
        'message': 'This endpoint will be implemented in Week 2'
    })


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500


# ==================== SERVER STARTUP ====================

if __name__ == '__main__':
    print("\n🚀 Starting server...")
    print("📍 URL: http://localhost:5000")
    print("✅ CORS enabled for Unity")
    print("\n📋 Available endpoints:")
    print("   GET  /health       - Check if server is running")
    print("   POST /test         - Test Unity communication")
    print("   POST /start_game   - (Coming in Week 2)")
    print("   POST /ask_question - (Coming in Week 2)")
    print("\n⏳ Press Ctrl+C to stop the server")
    print("=" * 50)
    print("\n")
    
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,       # Port number
        debug=True       # Enable debug mode (auto-reload on changes)
    )