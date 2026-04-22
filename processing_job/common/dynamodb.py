import json
import os
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError
from loguru import logger

AWS_REGION = os.environ["AWS_REGION"]


class DynamoDBOperations:
    """DynamoDB operations for pipeline step results.

    Schema:
        id  — input id from inputs.json (e.g. 'tokyo-rain-alley')
        step — composite sort key: '{step_name}#{model}#g{N}' (e.g. 't2v#ltx23#g0')
    """

    def __init__(self, table_name: str):
        try:
            self.dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
            self.table = self.dynamodb.Table(table_name)
            logger.info(f"Initialized DynamoDB operations for table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to initialize DynamoDB operations: {e}")
            raise

    @staticmethod
    def _serialize(data: dict) -> dict:
        """Convert dict values to DynamoDB-safe strings."""
        out = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                out[key] = json.dumps(value)
            else:
                out[key] = str(value)
        return out

    def put_item(self, id: str, step: str, data: dict) -> bool:
        """Write an item keyed by (id, step)."""
        try:
            item = {"id": id, "step": step, **self._serialize(data)}
            self.table.put_item(Item=item)
            logger.success(f"Put item id={id} step={step}")
            return True
        except ClientError as e:
            logger.error(f"DynamoDB ClientError: {e.response['Error']['Message']}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error writing item: {e}")
            return False

    @staticmethod
    def _deserialize_value(value):
        """Convert a DynamoDB value back to a Python type."""
        if isinstance(value, Decimal):
            return float(value) if value % 1 else int(value)
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        return value

    def get_item(self, id: str, step: str) -> dict:
        """Read an item by (id, step). Returns empty dict if not found."""
        try:
            resp = self.table.get_item(Key={"id": id, "step": step})
            if "Item" not in resp:
                logger.warning(f"No item found for id={id} step={step}")
                return {}
            return {k: self._deserialize_value(v) for k, v in resp["Item"].items()}
        except ClientError as e:
            logger.error(f"DynamoDB ClientError: {e.response['Error']['Message']}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error reading item: {e}")
            return {}

    def query_by_filename_prefix(self, filename_prefix: str) -> list[dict]:
        """Scan for items whose 'filename' attribute starts with the given prefix."""
        try:
            resp = self.table.scan(
                FilterExpression="begins_with(filename, :prefix)",
                ExpressionAttributeValues={":prefix": filename_prefix},
            )
            items = resp.get("Items", [])
            logger.info("Found {} items matching filename prefix '{}'", len(items), filename_prefix)
            return [{k: self._deserialize_value(v) for k, v in item.items()} for item in items]
        except ClientError as e:
            logger.error("DynamoDB ClientError: {}", e.response["Error"]["Message"])
            return []
        except Exception as e:
            logger.error("Unexpected error scanning: {}", e)
            return []

    def update_attribute(self, id: str, step: str, attr: str, value) -> bool:
        """Update a single attribute on an existing item."""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            else:
                value = str(value)
            self.table.update_item(
                Key={"id": id, "step": step},
                UpdateExpression="SET #a = :v",
                ExpressionAttributeNames={"#a": attr},
                ExpressionAttributeValues={":v": value},
            )
            logger.success(f"Updated {attr} on id={id} step={step}")
            return True
        except ClientError as e:
            logger.error(f"DynamoDB ClientError: {e.response['Error']['Message']}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating attribute: {e}")
            return False
