FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

RUN apt update && apt install -y nvidia-modprobe
RUN pip install adabelief-tf pandas imageio imageio-ffmpeg seaborn scikit-learn tensorflow_addons tqdm
RUN pip install keras_applications tensorflowjs
RUN pip install tf2onnx