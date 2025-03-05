import asyncio
import time
import yaml
from typing import Any
from vessl import vessl_api


async def wait_for_gateway_enabled(
    gateway: Any, service_name: str, max_timeout_sec: int = 8 * 60
) -> bool:
    """Waits for the gateway of a service to be enabled.

    Args:
        gateway (Any): The gateway configuration object.
        service_name (str): Name of the service.
        max_timeout_sec (int): Maximum timeout in seconds. Default is 8 minutes.

    Returns:
        bool: True if the gateway is successfully enabled, False otherwise.
    """

    def _check_gateway_enabled(gateway):
        return (
            gateway.enabled
            and gateway.status == "success"
            and gateway.endpoint is not None
        )

    if not _check_gateway_enabled(gateway):
        print("Endpoint update in progress. Please wait a moment.")
        print(">> Wait for the endpoint to be ready...")

        gateway_update_timeout = max_timeout_sec

        while gateway_update_timeout > 0:
            gateway = read_service(service_name=service_name).gateway_config
            if _check_gateway_enabled(gateway):
                break

            gateway_update_timeout -= 5
            time.sleep(5)
            print(
                f">> Waiting for {max_timeout_sec - gateway_update_timeout} seconds..."
            )

        if gateway_update_timeout <= 0:
            print("Endpoint update timeout. Please check the status of the endpoint.")
            return False

    return True


def read_service(service_name: str) -> Any:
    """Fetches the service configuration using the service name.

    Args:
        service_name (str): The name of the service.

    Returns:
        Any: The service configuration object.
        ```
    """
    return vessl_api.model_service_read_api(
        organization_name=vessl_api.organization.name, model_service_name=service_name
    )


def ensure_service_idempotence(service_name: str, yaml_str: str) -> Any:
    """Fetches the service configuration using the yaml string.

    Args:
        service_name (str): The name of the service.
        yaml_str (str): The string value of yaml.

    Returns:
        Any: The service configuration object.
    """
    resp = vessl_api.model_service_revision_list_api(
        organization_name=vessl_api.organization.name,
        model_service_name=service_name,
    )
    running_revision = None
    for r in resp.results:
        if r.status == "running":
            running_revision = r
            break
    if running_revision is None:
        return None

    if running_revision.status == "running":
        served_yaml_dict = yaml.safe_load(running_revision.yaml_spec)
        user_yaml_dict = yaml.safe_load(yaml_str)

        if all(
            [
                served_yaml_dict.get("env", {}).get("MODEL_NAME")
                == user_yaml_dict.get("env", {}).get("MODEL_NAME"),
                served_yaml_dict.get("run") == user_yaml_dict.get("run"),
                served_yaml_dict.get("image") == user_yaml_dict.get("image"),
            ]
        ):
            gateway = read_service(service_name=service_name).gateway_config
            return f"https://{gateway.endpoint}/v1"

    return None


def _request_abort_rollout(service_name: str) -> bool:
    """Requests to abort the ongoing rollout of a service.

    Args:
        service_name (str): The name of the service.

    Returns:
        bool: True if rollback is requested, False otherwise.
    """
    resp = vessl_api.model_service_rollout_abort_api(
        organization_name=vessl_api.organization.name,
        model_service_name=service_name,
    )
    if resp.rollback_requested:
        print("Current update aborted. Rollback is requested.\n")
        return True
    else:
        print("Current update aborted.")
        print("Could not determine the original status. Rollback is not requested.\n")
        print(
            f"Please check the status of the service and the gateway at: "
            f"https://app.vessl.ai/{vessl_api.organization.name}/services/{service_name}\n"
        )
        return False


def _get_recent_rollout(service_name: str) -> Any:
    """Fetches the most recent rollout information for a service.

    Args:
        service_name (str): The name of the service.

    Returns:
        Any: The most recent rollout object, or None if no rollouts are found.
    """
    resp = vessl_api.model_service_rollout_list_api(
        organization_name=vessl_api.organization.name,
        model_service_name=service_name,
    )
    return resp.rollouts[0] if resp.rollouts else None


async def abort_in_progress_rollout_by_name(service_name: str):
    """Aborts an ongoing rollout for a given service.

    Args:
        service_name (str): The name of the service.
    """
    recent_rollout = _get_recent_rollout(service_name)
    if recent_rollout and recent_rollout.status == "rolling_out":
        print(f"The service {service_name} is currently rolling out.")
        if _request_abort_rollout(service_name):
            print("Waiting for the existing rollout to be aborted...")
            await asyncio.sleep(30)
    else:
        print("No existing rollout found.")


# Note: The above functions are designed to work with the internal rules of the `vessl` package
