import os
import sys
from pathlib import Path

def create_file(filepath):
    """Create an empty file."""
    filepath.touch()
    print(f"Created file: {filepath}")

def create_init(directory):
    """Create __init__.py file in the directory."""
    init_file = directory / "__init__.py"
    create_file(init_file)

def create_project_structure(base_path):
    """Create the project directory structure."""
    # Convert string path to Path object
    base_dir = Path(base_path)

    # Create base project directory
    base_dir.mkdir(exist_ok=True)
    print(f"\nCreating project structure in: {base_dir.absolute()}\n")

    # Define the structure
    # structure = {
    #     "config": ["settings.py"],
    #     "src": {
    #         "api": ["kaggle_client.py", "datasets.py", "competitions.py", "kernels.py", "models.py", "files.py" ],
    #         "utils": ["helpers.py"],
    #         "handlers": ["data_handlers.py"]
    #     },
    #     "tests": ["test_kaggle_client.py"],
    #     "logs": [],
    #     "data": {
    #         "raw": [],
    #         "processed": []
    #     },
    #     "notebooks": []
    # }

    structure = {
        "operational_configs": {
                "dataset_configs":["datasets.yaml", "download_settings.yaml"],
                "file_configs": ["file_operations.yaml", "file_paths.yaml"],
                "model_configs": ["model_params.yaml", "training_config.yaml"],
                "kernel_configs": ["runtime_settings.yaml", "resource_limits.yaml"],
                "competition_configs": ["submission_rules.yaml", "competition_params.yaml"]
        },
        "workflows": {
            "dataset_workflows": ["__init__.py", "download_operations.py", "upload_operations.py"],
            "file_workflows": ["__init__.py", "download_operations.py", "upload_operations.py"],
            "model_workflows": ["__init__.py", "download_operations.py", "upload_operations"],
            "kernel_workflows": ["__init__.py", "execution_operations.py", "management_operations"],
            "competition_workflows": ["__init__.py", "submission_operations", "leaderboard_operations"]
        },
        "data": {
            "datasets": ["raw", "processed"],
            "files": ["downloads", "uploads", "backups"],
            "models": ["downloaded", "custom", "checkpoints"],
            "kernels": ["outputs", "scripts"],
            "competitions": ["submissions", "leaderboards"]
        }
    }

    # Create top-level files
    create_file(base_dir / "requirements.txt")
    create_file(base_dir / "README.md")
    create_file(base_dir / "main.py")
    create_file(base_dir / ".gitignore")

    # Create directories and their files
    for directory, contents in structure.items():
        dir_path = base_dir / directory
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")

        # If directory is under src, create __init__.py
        if directory == "src" or str(dir_path).endswith(('api', 'utils', 'handlers')):
            create_init(dir_path)

        # Create files in the directory
        if isinstance(contents, list):
            for file in contents:
                create_file(dir_path / file)
        else:
            # Handle nested directories
            for subdir, subcontents in contents.items():
                subdir_path = dir_path / subdir
                subdir_path.mkdir(exist_ok=True)
                print(f"Created directory: {subdir_path}")

                # Create files in subdirectory
                for file in subcontents:
                    create_file(subdir_path / file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_project.py <project_directory>")
        sys.exit(1)

    project_path = sys.argv[1]
    create_project_structure(project_path)
    print("\nProject structure created successfully!")
