from flask import Flask, jsonify, render_template, request, abort, current_app, make_response
from datetime import timedelta
from functools import update_wrapper
from flask_cors import CORS
app = Flask(__name__, static_path='/static')
CORS(app)
import os
import pickle
import subprocess

@app.route("/", methods=['GET'])
def askApi():
    return render_template("index.html")

@app.route("/api/v2.0/<string:text>", methods=['GET'])
def api(text):
    newText = text.lower()

    try:
        new = subprocess.check_output(["python3", "saver.py", text])

    except IndexError:
        abort(404)
    
    return jsonify({'analyze': str(new)})