"""ComfyUI queue status checker.

Provides ``get_queue_size()`` for direct import and a CLI entry point.
"""

import requests
from loguru import logger
from requests.exceptions import RequestException

ENDPOINT = "http://127.0.0.1:8188/prompt"


def get_queue_size() -> int | None:
    """Return the number of remaining items in the ComfyUI queue, or None on error."""
    try:
        response = requests.get(ENDPOINT, timeout=10)
        response.raise_for_status()
        data = response.json()
        queue_size = data["exec_info"]["queue_remaining"]
        logger.info("Queue size: {}", queue_size)
        return queue_size
    except requests.exceptions.HTTPError as err:
        logger.error("HTTP Error: {}", err)
        logger.error("Response content: {}", err.response.text)
    except requests.exceptions.ConnectionError:
        logger.error("Connection Error: Failed to connect to the server. Is the server running?")
    except requests.exceptions.Timeout:
        logger.error("Timeout Error: The request timed out")
    except RequestException as e:
        logger.error("Error: {}", e)
    return None


if __name__ == "__main__":
    get_queue_size()
