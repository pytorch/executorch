Dependancies for Deepfilternet and Eencodec

pip install -q -U deepfilternet
pip install -U encodec  # stable release
pip install -U git+https://git@github.com/facebookresearch/encodec#egg=encodec


Steps for running encodec.py and deepfilternet.py

Note: 1. The input used for the both models are random input. and hardcoded and being used as the tensor directly
        2. The files deepfilternet.py and encodec.py should be available at location executorch/examples/cadence/models


cd executorch 
python3 -m examples.cadence.models.deepfilternet |& tee examples/cadence/models/log_deepfilternet.log
python3 -m examples.cadence.models.encodec |& tee examples/cadence/models/log_encodec.log
