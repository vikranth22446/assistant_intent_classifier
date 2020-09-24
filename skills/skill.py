from enum import Enum, auto
import paramiko
import zipfile
import gdown
from abc import ABC, abstractmethod 
from flask import Flask, jsonify


class DownloadTypes:
    Google = auto()
    SFTP = auto()
    Docker = auto()
    URL = auto()

class DownloadPath:
    def __init__(file_name, download_loc, download_type, download_args, is_zip=False):
        self.file_name = file_name
        self.download_loc = download_loc
        self.download_type = download_type
        self.download_args = download_args
        self.is_zip = is_zip

class RunConfig:
    def __init__(docker_image=None, docker_args=None, local_server=None, host=None, port=None)
        self.docker_image = docker_image
        self.docker_args = docker_args
        self.local_server = local_server
        self.server_endpoint = server_endpoint
        self.host = host
        self.port = port

class Skill(ABC):
    def __init__(run_config, model_files_download_paths=[], model_folder=None):
        self.run_config = run_config
        self.model_folder = model_folder
        if not model_folder:
            self.model_folder = "./"
        self.model_files_download_paths = model_files_download_paths

    @abstractmethod
    def init_models(self):
        pass

    def download_models(self):
        for model_file in model_files_download_paths:
            download_loc = os.path.join(self.model_folder, model_file.download_name)
            full_file_name = model_file.file_name
            if model_file.is_zip:
                full_file_name = f"{file_name}.zip"
            if os.path.exists(download_loc):
                continue 
            if model_file.download_type == DownloadTypes.Google:
                url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vNm9zMTJWOGxobkU'
                gdown.download(model_file.download_loc, full_file_name, quiet=False)
            elif model_file.download_type == DownloadTypes.SFTP:
                ssh,sft = get_ssh_sftp_instance()
                sftp.get(model_file.download_loc, full_file_name)
                cleanup_ssh_sftp(ssh, sftp)
            elif model_file.download_type == DownloadTypes.URL:
                res = requests.get(model_file.download_loc, stream=True)
                file_size = int(res.headers.get("Content-Length", 0))
                progress = tqdm(res.iter_content(buffer_size), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
                with open(full_file_name, "wb") as f:
                    for data in progress:
                        f.write(data)
                        progress.update(len(data))
            if model_file.iszip:
                with zipfile.ZipFile(full_file_name,"r") as zip_ref:
                    zip_ref.extractall(model_file.file_name)

    def run_skill_locally(self):
        app = Flask(__name__)
        @app.route('/classify', methods=["POST", "GET"])
        @app.route('/classify/<text>', methods=["POST", "GET"])
        def classify_endpoint(text=None):
            if not text:
                json_req = request.get_json()
                text = json_req.text
            prob, label = self.classify(text)
            return jsonify({"prob": prob, "label": label})


def get_ssh_sftp_instance(SSH_CONFIG):
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	ssh.load_system_host_keys()
	ssh.connect(SSH_CONFIG["host"], username=SSH_CONFIG["username"])
	sftp = ssh.open_sftp()

	print("Connected to SSH, SFTP Client")
	return ssh, sftp

def cleanup_ssh_sftp(sftp, ssh):
	sftp.close()
	ssh.close()
