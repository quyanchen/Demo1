from pathlib import Path

import swanlab

def config_to_dict(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

def init_swanlab(args):
    return swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        config=config_to_dict(args),
        log_dir=str(args.output_dir / "swanlog"),
        mode=args.swanlab_mode,
    )

def log_metrics(run, split, metrics, step):
    log_data = {f"{split}/{k}": v for k, v in metrics.items()}
    run.log(log_data, step=step)
