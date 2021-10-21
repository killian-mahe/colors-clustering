FROM python:3.9

WORKDIR /user/src/app

RUN python3 -m venv /opt/venv

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libopengl0 libegl1  -y

# Install dependencies:
COPY requirements.txt .
RUN . /opt/venv/bin/activate && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ls

CMD . /opt/venv/bin/activate && exec python colors_clustering