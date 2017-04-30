python3 does not have mysql drivers installed

Do the following to setup the environment

virtualenv venv -p python2
source venv/bin/activate
pip install --upgrade pip
pip install --trusted-host pypi.python.org -r requirements.txt

Michale Collins parser for extracting subject/object/neither from here http://www.cs.columbia.edu/~mcollins/code.html


