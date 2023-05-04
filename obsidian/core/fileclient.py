import json
import yaml
import os


def read_file(filename):
    if not os.path.isfile(filename):
        raise ValueError(f"No such file: {filename}")
    file_client = FileClient(filename)
    return file_client.read_file()


class FileClient:

    def __init__(self, filename):
        self.filename = filename
        self.file_ext = self.get_file_extension(filename)

    def get_file_extension(self, filename):
        return filename.split(".")[-1]

    def read_file(self):
        if self.file_ext == "json":
            with open(self.filename, "r") as f:
                return json.load(f)
        elif self.file_ext == "yaml" or self.file_ext == "yml":
            with open(self.filename, "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError(
                "Invalid file extension. Only JSON and YAML files are supported.")


if __name__ == '__main__':
    contents = read_file('configs/detection/base.yaml')
    print(contents)
