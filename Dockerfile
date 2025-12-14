# 1. Base Image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 2. Munkakönyvtár
WORKDIR /workspace

# 3. Rendszerfüggőségek
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Python csomagok telepítése (requirements.txt-ből)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. run.sh script bemásolása (ha van)
COPY ./src/run.sh .
RUN chmod +x run.sh

# 6. Jupyter alapbeállítások – token és jelszó kikapcsolása
RUN jupyter server --generate-config && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.allow_origin = '*'" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.disable_check_xsrf = True" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.NotebookApp.notebook_dir = '/workspace'" >> /root/.jupyter/jupyter_server_config.py

# 7. Port megnyitása
EXPOSE 8888

# 8. Indítási parancs
CMD ["bash", "run.sh"]
