"""Docker Code Interpreter tool spec for code execution."""

import docker
import tempfile
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from llama_index.core.tools.tool_spec.base import BaseToolSpec


def _sanitize_input(query: str) -> str:
    """Sanitize input and remove whitespace, backtick, and markdown.

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """
    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    # Add new line if no new line was appended at the end of the query
    if not query.endswith("\n"):
        query += "\n"
    return query


class DockerCodeToolSpec(BaseToolSpec):
    """Docker Code Interpreter tool spec.

    Leverages Docker to execute Python code and persist the workspace.
    """

    spec_functions = [
        "execute_code",
        "execute_file",
        "list_files",
        "write_file",
        "create_directory",
    ]

    def __init__(
        self,
        base_image: str = "python:3.9-slim",
        requirements: Optional[Union[List[str], str]] = None,
        memory_limit: str = "100m",
        max_processes: int = 10,
        workspace_dir: Optional[str] = None,
        dockerfile: Optional[str] = None,
        build_args: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize a persistent Docker session executor with custom dependencies.

        Args:
            base_image: Base Docker image to use
            requirements: List of pip packages or path to requirements.txt
            memory_limit: Maximum memory usage
            max_processes: Maximum number of processes allowed
            workspace_dir: Optional persistent directory for workspace
            dockerfile: Optional custom Dockerfile path
            build_args: Optional build arguments for Dockerfile
            session_id: Optional unique identifier for the session
        """
        self.client = docker.DockerClient()
        self.memory_limit = memory_limit
        self.max_processes = max_processes

        # Set up workspace directory
        if workspace_dir:
            self.workspace_root = Path(workspace_dir)
        else:
            self.workspace_root = Path(tempfile.mkdtemp(prefix="docker_sessions_"))
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Build custom image with dependencies if needed
        if requirements is None and dockerfile is None and build_args is None:
            self.image_name = base_image
        else:
            self.image_name = self._build_custom_image(
                base_image=base_image,
                requirements=requirements,
                dockerfile=dockerfile,
                build_args=build_args,
            )

        self.workspace = None
        self.container = None
        self.session_id = self._create_session(session_id=session_id)

    def _build_custom_image(
        self,
        base_image: str,
        requirements: Optional[Union[List[str], str]],
        dockerfile: Optional[str],
        build_args: Optional[Dict[str, str]],
    ) -> str:
        """Build custom Docker image with dependencies."""
        image_tag = f"custom_python_env_{uuid.uuid4().hex[:8]}"

        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)

            if dockerfile:
                # Use custom Dockerfile
                shutil.copy(dockerfile, build_path / "Dockerfile")
            else:
                # Generate Dockerfile with requirements
                dockerfile_content = [
                    f"FROM {base_image}",
                    "WORKDIR /workspace",
                    # Install system dependencies
                    "RUN apt-get update && apt-get install -y --no-install-recommends \\\n",
                    "    build-essential \\\n",
                    "    git \\\n",
                    "    && rm -rf /var/lib/apt/lists/*",
                    # Upgrade pip
                    "RUN pip install --no-cache-dir --upgrade pip",
                ]

                # Handle requirements
                if requirements:
                    if isinstance(requirements, str) and os.path.isfile(requirements):
                        # Copy requirements.txt
                        shutil.copy(requirements, build_path / "requirements.txt")
                        dockerfile_content.append(
                            "COPY requirements.txt /workspace/requirements.txt\n"
                            "RUN pip install --no-cache-dir -r requirements.txt"
                        )
                    elif isinstance(requirements, (list, tuple)):
                        # Install from list
                        requirements_str = " ".join(requirements)
                        dockerfile_content.append(
                            f"RUN pip install --no-cache-dir {requirements_str}"
                        )

                # Write Dockerfile
                dockerfile_path = build_path / "Dockerfile"
                dockerfile_path.write_text("\n".join(dockerfile_content))

            # Build the image
            try:
                image, logs = self.client.images.build(
                    path=str(build_path), tag=image_tag, buildargs=build_args, rm=True
                )

                return image_tag

            except Exception as e:
                raise RuntimeError(f"Failed to build custom image: {e!s}")

    def _create_session(
        self,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Create a new persistent Docker session.

        Args:
            session_id: Optional unique identifier for the session

        Returns:
            session_id: Unique identifier for the session
        """
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        # Create session workspace
        workspace = self.workspace_root / session_id
        workspace.mkdir(parents=True, exist_ok=True)

        # Start container
        self.container = self.client.containers.run(
            self.image_name,
            command="tail -f /dev/null",  # Keep container running
            name=f"session_{session_id}",
            detach=True,
            working_dir="/workspace",
            volumes={str(workspace.absolute()): {"bind": "/workspace", "mode": "rw"}},
            mem_limit=self.memory_limit,
            pids_limit=self.max_processes,
            network_mode="none",
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
        )
        self.workspace = workspace

        return session_id

    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in a specific session.

        Args:
            code (str): Python code to execute

        Returns:
            Dict containing execution results
        """
        # Write code to a temporary file in the workspace
        code = _sanitize_input(code)
        code_file = str(uuid.uuid4())[:8] + ".py"
        exec_cmd = f"python -c '{code}'"

        try:
            exit_code, output = self.container.exec_run(
                cmd=["sh", "-c", exec_cmd],
                workdir="/workspace",
                demux=True,  # Split stdout/stderr
            )

            stdout = output[0].decode("utf-8") if output[0] else ""
            stderr = output[1].decode("utf-8") if output[1] else ""

            return {
                "success": exit_code == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            }

        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Error during execution: {e!s}",
                "exit_code": -1,
            }

    def execute_file(self, filename: str) -> Dict[str, Any]:
        """
        Execute a Python file in the session workspace.

        Args:
            filename (str): Name of the file to execute

        Returns:
            Dict containing execution results
        """
        filepath = Path("/workspace") / filename

        try:
            exit_code, output = self.container.exec_run(
                cmd=["python", str(filepath)], workdir="/workspace", demux=True
            )

            stdout = output[0].decode("utf-8") if output[0] else ""
            stderr = output[1].decode("utf-8") if output[1] else ""

            return {
                "success": exit_code == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            }

        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Error during execution: {e!s}",
                "exit_code": -1,
            }

    def list_files(self, path: str = ".") -> List[str]:
        """
        List files in the session workspace.

        Args:
            path (str): Relative path within workspace

        Returns:
            List of files and directories
        """
        exit_code, output = self.container.exec_run(
            cmd=["ls", "-la", path], workdir="/workspace"
        )

        if exit_code == 0:
            return output.decode("utf-8").splitlines()
        else:
            raise RuntimeError(f"Failed to list files: {output.decode('utf-8')}")

    def write_file(self, filename: str, content: str) -> bool:
        """
        Write content to a file in the session workspace.

        Args:
            session_id: Session identifier
            filename (str): Name of the file to write
            content (str): Content to write to the file

        Returns:
            bool indicating success
        """
        filepath = self.workspace / filename

        try:
            filepath.write_text(content)
            return True
        except Exception as e:
            return False

    def create_directory(self, dirname: str) -> bool:
        """
        Create a directory in the session workspace.

        Args:
            dirname (str): Name of the directory to create

        Returns:
            bool indicating success
        """
        exit_code, output = self.container.exec_run(
            cmd=["mkdir", "-p", dirname], workdir="/workspace"
        )

        return exit_code == 0

    def __del__(self) -> None:
        """Clean up resources when the object is deleted."""
        self.container.stop()
        self.client.close()
