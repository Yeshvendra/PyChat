#######################################################################################################
# FILE DESCRIPTION
# This file contains server code to host the PyChat Application.
#######################################################################################################

from flask import Flask, render_template, request
from PyChat_Core_Code import GenerateResponse

app = Flask(__name__)

# Create endpoint to host PyChat HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Create endpoint to cater input query of PyChat
@app.route('/submit', methods=["POST"])
def processInputQuery():
    req = request.get_json(silent = True, force = True)
    reponse, confident = GenerateResponse(req["msg"])
    return reponse

# Main to run the server on specific port
if __name__ == '__main__':
    app.run(debug = True, port = 2412)
