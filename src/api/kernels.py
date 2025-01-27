from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from src.utils.error_handlers import handle_api_errors, validate_auth
from .kaggle_client import KaggleAPIClient

@dataclass
class KernelMetadata:
    title: str
    language: str
    kernel_type: str
    is_private: bool = False
    enable_gpu: bool = False
    enable_internet: bool = False
    dataset_sources: Optional[List[str]] = None
    competition_sources: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

class KernelClient:
    def __init__(self, client: KaggleAPIClient):
        self.client = client

    @handle_api_errors
    @validate_auth
    def list_kernels(
        self,
        owner: Optional[str] = None,
        language: Optional[str] = None,
        page: int = 1,
        **kwargs
    ) -> List[Dict]:
        params = {'owner': owner, 'language': language, 'page': page, **kwargs}
        response = self.client.get('kernels/list', params=params)
        return response.json()

    @handle_api_errors
    @validate_auth
    def push_kernel(
        self,
        folder_path: Union[str, Path],
        metadata: KernelMetadata,
        version_notes: Optional[str] = None
    ) -> Dict:
        result = self.client.post('kernels/push', json=metadata.to_dict()).json()

        folder_path = Path(folder_path)
        for file_path in folder_path.glob('**/*'):
            if file_path.is_file():
                self.client.post(
                    'kernels/push/file',
                    files={'file': open(file_path, 'rb')},
                    data={'path': str(file_path.relative_to(folder_path))}
                )

        if version_notes:
            self.client.post(
                'kernels/push/version',
                json={'notes': version_notes}
            )

        return result

    @handle_api_errors
    @validate_auth
    def pull_kernel(
        self,
        owner: str,
        kernel_name: str,
        version: Optional[str] = None,
        path: Optional[Union[str, Path]] = None
    ) -> Path:
        path = Path(path) if path else Path.cwd()
        path.mkdir(parents=True, exist_ok=True)

        params = {
            'ownerSlug': owner,
            'kernelSlug': kernel_name,
            'version': version
        }
        response = self.client.get('kernels/pull', params=params)
        kernel_files = response.json()

        for file_info in kernel_files:
            file_path = path / file_info['path']
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file_info['source'])

        return path

    @handle_api_errors
    @validate_auth
    def get_kernel_status(
        self,
        owner: str,
        kernel_name: str,
        version: Optional[str] = None
    ) -> Dict:
        params = {
            'ownerSlug': owner,
            'kernelSlug': kernel_name,
            'version': version
        }
        return self.client.get('kernels/status', params=params).json()

    def wait_for_kernel_output(
        self,
        owner: str,
        kernel_name: str,
        timeout: int = 3600,
        check_interval: int = 10
    ) -> Dict:
        import time
        start_time = time.time()

        while True:
            status = self.get_kernel_status(owner, kernel_name)

            if status.get('status') == 'complete':
                response = self.client.get(
                    'kernels/output',
                    params={
                        'ownerSlug': owner,
                        'kernelSlug': kernel_name
                    }
                )
                return response.json()

            if status.get('status') == 'error':
                raise RuntimeError(f"Kernel failed: {status.get('errorMessage')}")

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Kernel not completed after {timeout} seconds")

            time.sleep(check_interval)

    @handle_api_errors
    @validate_auth
    def list_kernel_versions(
        self,
        owner: str,
        kernel_name: str
    ) -> List[Dict]:
        params = {
            'ownerSlug': owner,
            'kernelSlug': kernel_name
        }
        return self.client.get('kernels/list/versions', params=params).json()
