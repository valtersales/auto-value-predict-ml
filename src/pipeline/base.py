"""
Base classes for pipeline steps.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class PipelineStep(ABC):
    """
    Abstract base class for pipeline steps.
    
    Each step in the pipeline should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize a pipeline step.
        
        Args:
            name: Name of the step
            enabled: Whether this step is enabled (default: True)
        """
        self.name = name
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline step.
        
        Args:
            context: Dictionary containing data and state from previous steps
                    Common keys:
                    - 'data': DataFrame or data from previous step
                    - 'config': Configuration dictionary
                    - 'artifacts': Dictionary to store artifacts (models, etc.)
        
        Returns:
            Updated context dictionary
        """
        pass
    
    @abstractmethod
    def validate(self, context: Dict[str, Any]) -> bool:
        """
        Validate that prerequisites for this step are met.
        
        Args:
            context: Current pipeline context
        
        Returns:
            True if validation passes, False otherwise
        """
        pass
    
    def get_dependencies(self) -> List[str]:
        """
        Get list of step names that this step depends on.
        
        Returns:
            List of step names
        """
        return []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"

