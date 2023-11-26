REM Create a Virtual Environment
python -m venv venv
venv\Scripts\activate
REM Install Python Wheel
pip install wheel
REM Create the actorcritic package
pip install -e .\src
REM Install requirements
pip install -r requirements.txt