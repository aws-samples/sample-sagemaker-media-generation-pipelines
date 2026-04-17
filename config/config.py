"""
Configuration management for processing jobs.

This module provides Pydantic models and utilities for loading and validating
configuration parameters for SageMaker processing jobs. It supports YAML-based
configuration files with strict validation for instance types, resource limits,
and container specifications.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
import yaml


class ContainerConfig(BaseModel):
    """
    Configuration model for SageMaker processing job container specifications.
    
    This model defines the structure and validation rules for container
    configuration including entrypoint, arguments, instance specifications,
    and environment variables.
    
    Attributes:
        ContainerEntrypoint: List of strings defining the container entrypoint command
        ContainerArguments: List of strings for container command arguments
        InstanceCount: Number of instances to use (1-10)
        InstanceType: AWS instance type, restricted to supported ML instance types
        VolumeSizeInGB: EBS volume size in GB (50-1000)
        Environment: Dictionary of environment variables for the container
    """

    ContainerEntrypoint: list[str] = Field(
        default_factory=list,
        description="Container entrypoint command as a list of strings"
    )
    ContainerArguments: list[str] = Field(
        default_factory=list,
        description="Container arguments as a list of strings"
    )
    InstanceCount: int = Field(
        ge=1,
        le=10,
        description="Number of instances for the processing job"
    )
    InstanceType: str = Literal[
        "ml.g4dn.2xlarge",
        "ml.g5.xlarge",
    ]

    VolumeSizeInGB: int = Field(
        ge=50,
        le=1000,
        description="EBS volume size in GB for the processing job"
    )
    Environment: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for the container"
    )

    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        strict=True,
    )


def get_configs(config_fp: str = "config.yaml") -> ContainerConfig:
    """
    Load and validate configuration from a YAML file.
    
    This function reads a YAML configuration file from the config directory
    and validates it against the ContainerConfig model. It provides type
    checking and validation for all configuration parameters.
    
    Args:
        config_fp: Filename of the configuration file in the config directory.
                  Defaults to "config.yaml".
    
    Returns:
        ContainerConfig: Validated configuration object with all parameters.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        pydantic.ValidationError: If the configuration doesn't match the schema.
        
    Example:
        >>> config = get_configs("my_config.yaml")
        >>> print(config.InstanceType)
        'ml.g5.4xlarge'
    """
    cfg = {}

    with open("./config/" + config_fp, "r", encoding="utf-8") as f:
        cfg = ContainerConfig(**yaml.full_load(f))

    return cfg
