import requests
from requests.exceptions import RequestException

ENDPOINT = "http://127.0.0.1:8188/prompt"

def main():
    try:
        response = requests.get(ENDPOINT, timeout=10)
        response.raise_for_status()

        data = response.json()
        queue_size = data['exec_info']['queue_remaining']
        print(f"Queue size: {queue_size}")
        return queue_size

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
        print(f"Response content: {err.response.text}")
    except requests.exceptions.ConnectionError:
        print("Connection Error: Failed to connect to the server. Is the server running?")
    except requests.exceptions.Timeout:
        print("Timeout Error: The request timed out")
    except RequestException as e:
        print(f"Error: {e}")
    return None


if __name__ == "__main__":
    main()

