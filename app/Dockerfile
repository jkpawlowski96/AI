FROM continuumio/miniconda:latest

WORKDIR /home/docker_conda_template

COPY environment.yml ./
RUN conda env create -f environment.yml

RUN echo "source activate ai" > ~/.bashrc

ENV PATH /opt/conda/envs/ai/bin:$PATH

COPY . ./app
COPY run.py ./

EXPOSE 5000

ENTRYPOINT  ["python3"]

CMD ["run.py"]