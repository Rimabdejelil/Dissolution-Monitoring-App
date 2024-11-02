from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('my event')
def emit_custom_message():
    message = 'Hello from Flask!'
    socketio.emit('my event', {'data': message})

@app.route('/emit')
def emit_event():
    emit_custom_message()
    return 'Event emitted!'

if __name__ == '__main__':
    socketio.run(app, debug=True)
