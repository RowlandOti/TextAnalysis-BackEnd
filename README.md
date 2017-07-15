# TextAnalysis-BackEnd
:love_letter: Back end for text analysis

### Flask API For TextAnalysis

The BackEnd!
We'll use python3 in a virtualenv, ensure you have it installed

```sh
virtualenv -p python3 TextEnv
cd TextEnv

source bin/activate

```
We need a few Dependencies: PS ensure you have activated the virtualenv

```sh
pip3 install flask
pip3 install flask-cors
pip3 install cython
pip3 install numpy
```

Then clone the TextAnalysis-BackEnd. 

```sh
cd TextAnalysis-BackEnd

export FLASK_APP=allinone.python3
flask run
```

Head over to your browser and rock on!