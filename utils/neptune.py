import os


def init_neptune(project: str = None, api_token: str = None, name: str = None):
    status = False
    neptune = None
    if project is not None and api_token is not None:
        try:
            import neptune.new as neptune

            os.environ["NEPTUNE_PROJECT"] = project
            os.environ["NEPTUNE_API_TOKEN"] = api_token
            if name is not None:
                os.environ["NEPTUNE_RUN_NAME"] = name

            status = True
        except ImportError:
            print("neptune not installed, skipping neptune logging. 'pip install neptune-client' for neptune logging.")
        except Exception as e:
            print(e)

    return status, neptune


def log_to_neptune(neptune_run, dict):
    for k, v in dict.items():
        neptune_run[k].log(v)
