#create a backend using flask
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/models/v1', methods=['POST'])
def api():
    if request.method == 'POST':
        return jsonify("api endpoint hit!")
    else:
        return jsonify("404 Server Error!")
    
if __name__ == '__main__':
    app.run(port=5000, debug=True)