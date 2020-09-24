# Assistant Intent Classifier
## Code Structure
```
├── Makefile
    - Common useful commands
├── app
    - Contains server code for Web interface 
├── notebooks
- Jupyter notebooks used to create/train the models
├── server.py
-  Run the Web Interface server locally
├── skills
- Skills created such as shopping and time
├── transcription
- transcription via deepspeech 
```
## Local Setup Instructions

Install the requirements via
`pip3 install -r requirements.txt` and run the jupyter notebook

and run server via
```python
pip3 server.py
```
## Docker Setup Instructions

Use The Makefile to run 
``
make run-core
``


## Architecture Documentation
One Architecture Diagram: https://people.eecs.berkeley.edu/~nmalkin/alva/architecture

Another Design/Diagram

![Architecture Diagram](images/blues_arch_diagram.png)