# TextAnalysis-BackEnd
:love_letter: Back end for text analysis

### Flask API For TextAnalysis

On the DIR with the file 'controller.py' Run

We all dont like flask running on port 5000 so lets install gunicorn and do it like the 'pros' :)

```sh
pip install gunicorn
```
Then, on the folder with allinone.py
```sh
export FLASK_APP=allinone.py

gunicorn allinone:app
```

It should start the server on your machine.

:)