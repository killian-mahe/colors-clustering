FROM python:3.9

WORKDIR /user/src/app

RUN python3 -m venv /opt/venv

# Install dependencies:
COPY requirements.txt .
RUN . /opt/venv/bin/activate && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "colors_clustering"]