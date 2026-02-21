"""
Base Agent Class
All agents inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from loguru import logger


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        logger.info(f"Initialized {agent_name}")

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary function"""

    def log_execution(self, message: str):
        """Log agent activity"""
        logger.info(f"[{self.agent_name}] {message}")
