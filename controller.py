from flask import Flask, jsonify, render_template, request, abort, current_app, make_response
from datetime import timedelta
from functools import update_wrapper
app = Flask(__name__, static_path='/static')
import os

# had to use this lengthy method for my CORS, but i discovered flask_cors, will implement in next push :)

def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator



# an idea of what your api would look like
analyze = [
    {
        'id' : 1,
        'key': 'hi',
        'sentiment': u'Greeting',
        'ratings': 0
    }, 
    {   
        'id': 2,
        'key': 'bye',
        'sentiment': u'See you later',
        'ratings': 0
    }]


@app.route("/", methods=['GET'])
def askApi():
    #return render_template("index.html")
    return jsonify({'analyze': analyze})


@app.route("/api/v1.0/<string:text>", methods=['GET'])
@crossdomain(origin='*')
def api(text):
    newText = text.lower()

    try:
        analyzed = [sent for sent in analyze if sent['key'] == newText ]
    
    except IndexError:
        return ""

    if len(newText) == 0:
        abort(404)

    #return jsonify({'analyze': analyzed[0]})
    return "Your Text %s analyzed is %s" %(sent['key'], sent['sentiment'])


@app.route("/api/v2.0/<string:text>", methods=['GET'])
@crossdomain(origin='*')
def index(text):
    newText = text.lower()

    try:
        analyzed = [sent for sent in analyze if sent['key'] == newText ]

    except IndexError:
        abort(404)
    return jsonify({'analyze': analyzed[0]})


# still buggy will fix later

"""@app.route("/api/v2.0/newSentiment", methods=["POST"])
@crossdomain(origin='*')
def newSents():
    if not request.json or not 'key' in request.json:
        abort(400)

    newSent = {
        'id': analyze[-1]['id'] +1,
        'key': request.json['key'],
        #'sentiment': request.json['sentiment']
    }

    analyze.append(newSent)
    return jsonify({'analyze': newSent}), 201"""