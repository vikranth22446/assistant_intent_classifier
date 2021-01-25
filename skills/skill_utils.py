import os
import zipfile
from abc import ABC, abstractmethod
from enum import auto
from multiprocessing.context import Process

import gdown
import paramiko
import requests
import wget
from flask import Flask, jsonify, request


class DownloadTypes:
    Google = auto()
    SFTP = auto()
    Docker = auto()
    URL = auto()


class DownloadPath:
    def __init__(
        self,
        file_name,
        download_loc,
        download_type,
        download_args,
        is_zip=False,
        ssh_config=None,
    ):
        self.file_name = file_name
        self.download_loc = download_loc
        self.download_type = download_type
        self.download_args = download_args
        self.is_zip = is_zip
        self.ssh_config = ssh_config


class RunConfig:
    def __init__(
        self,
        docker_image=None,
        docker_args=None,
        local_server=None,
        host=None,
        port=None,
        server_endpoint=None,
    ):
        self.docker_image = docker_image
        self.docker_args = docker_args
        self.local_server = local_server
        self.server_endpoint = server_endpoint
        self.host = host
        self.port = port


class Skill(ABC):
    skill_name = None
    
    def __init__(self, run_config, model_files_download_paths=None, model_folder=None):
        if model_files_download_paths is None:
            model_files_download_paths = []
        self.run_config = run_config
        self.model_folder = model_folder
        if not model_folder:
            self.model_folder = "./"
        self.model_files_download_paths = model_files_download_paths
        self.running_locally = None

    @abstractmethod
    def init_models(self):
        pass

    def download_models(self):
        for model_file in self.model_files_download_paths:
            download_loc = os.path.join(self.model_folder, model_file.download_name)
            full_file_name = model_file.file_name
            if os.path.exists(download_loc):
                continue
            if model_file.download_type == DownloadTypes.Google:
                gdown.download(model_file.download_loc, full_file_name, quiet=False)
            elif model_file.download_type == DownloadTypes.SFTP:
                ssh, sftp = get_ssh_sftp_instance(model_file.ssh_config)
                sftp.get(model_file.download_loc, full_file_name)
                cleanup_ssh_sftp(ssh, sftp)
            elif model_file.download_type == DownloadTypes.URL:
                wget.download(model_file.download_loc)
            if model_file.iszip:
                with zipfile.ZipFile(full_file_name, "r") as zip_ref:
                    zip_ref.extractall(model_file.file_name)

    def handle_classification(self):
        if self.run_config.local_server:
            if not self.run_config.running_locally:
                p = Process(target=self.run_skill_locally, args=(self,))
                p.start()
                self.run_config.running_locally = p

            requests.get(
                "http://{}:{}".format(self.run_config.host, self.run_config.port)
            )

    def cleanup(self):
        if self.run_config.running_locally:
            self.running_locally.kill()

    def run_skill_locally(self):
        app = Flask(__name__)

        @app.route("/classify", methods=["POST", "GET"])
        @app.route("/classify/<text>", methods=["POST", "GET"])
        def classify_endpoint(text=None):
            if not text:
                json_req = request.get_json()
                text = json_req.text
            prob, label = self.classify(text)
            return jsonify({"prob": prob, "label": label})

        app.run(self.run_config.host, self.run_config.port)

    @abstractmethod
    def classify(self, text):
        pass


def get_ssh_sftp_instance(ssh_config):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_system_host_keys()
    ssh.connect(ssh_config["host"], username=ssh_config["username"])
    sftp = ssh.open_sftp()
    return ssh, sftp


def cleanup_ssh_sftp(sftp, ssh):
    sftp.close()
    ssh.close()
