#create a backend using flask
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/models/v1', methods=['GET'])
def api():
    if request.method == 'GET':
        return jsonify("api endpoint hit!!")
    else:
        return jsonify("404 Server Error!")
    
if __name__ == "__main__":
    app.run(host=os.environ.get('HOST', '0.0.0.0'),
            port=int(os.environ.get('PORT', 5000)))