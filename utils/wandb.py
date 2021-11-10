def init_wandb(project: str = None, name: str = None, id: str = None):
    status = False
    wandb = None
    if project is not None and name is not None:
        try:
            import wandb

            if not id:
                wandb.init(project=project, name=name)
            else:
                wandb.init(project=project, name=name, id=id)
            status = True
        except ImportError:
            print("wandb not installed, skipping wandb logging. 'pip install wandb' for wandb logging.")
        except Exception as e:
            print(e)

    return status, wandb


def log_to_wandb(wandb, dict):
    wandb.log({**dict})
