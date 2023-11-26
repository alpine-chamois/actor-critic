# Create a Virtual Environment
python -m venv venv
. venv/bin/activate
# Install Python Wheel
pip install wheel
# Create the actorcritic package
pip install -e ./src
# Install requirements
pip install -r requirements.txt