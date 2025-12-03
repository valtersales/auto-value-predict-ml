"""
Main pipeline orchestrator for AutoValuePredict ML project.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime

from .base import PipelineStep

logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Main ML pipeline orchestrator.
    
    Manages execution of pipeline steps in order, handles dependencies,
    and provides logging and state management.
    """
    
    def __init__(
        self,
        name: str = "AutoValuePredict Pipeline",
        output_dir: Optional[str] = None,
        save_state: bool = True
    ):
        """
        Initialize the ML pipeline.
        
        Args:
            name: Name of the pipeline
            output_dir: Directory to save pipeline artifacts (default: data/processed)
            save_state: Whether to save pipeline state after each step
        """
        self.name = name
        self.steps: List[PipelineStep] = []
        
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            self.output_dir = project_root / "data" / "processed"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.context: Dict[str, Any] = {
            'config': {},
            'artifacts': {},
            'output_dir': str(self.output_dir),
            'metadata': {
                'pipeline_name': name,
                'created_at': datetime.now().isoformat(),
                'steps_executed': []
            }
        }
        
        self.save_state = save_state
    
    def add_step(self, step: PipelineStep) -> 'MLPipeline':
        """
        Add a step to the pipeline.
        
        Args:
            step: PipelineStep instance to add
        
        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        logger.info(f"Added step: {step.name}")
        return self
    
    def add_steps(self, steps: List[PipelineStep]) -> 'MLPipeline':
        """
        Add multiple steps to the pipeline.
        
        Args:
            steps: List of PipelineStep instances
        
        Returns:
            Self for method chaining
        """
        for step in steps:
            self.add_step(step)
        return self
    
    def _validate_dependencies(self, step: PipelineStep) -> bool:
        """
        Validate that all dependencies for a step are satisfied.
        
        Args:
            step: Step to validate
        
        Returns:
            True if all dependencies are satisfied
        """
        dependencies = step.get_dependencies()
        executed_steps = self.context['metadata']['steps_executed']
        
        for dep in dependencies:
            if dep not in executed_steps:
                logger.error(
                    f"Step '{step.name}' requires dependency '{dep}' "
                    f"which has not been executed yet."
                )
                return False
        
        return True
    
    def _save_state(self):
        """Save pipeline state to disk."""
        if not self.save_state:
            return
        
        state_file = self.output_dir / "pipeline_state.json"
        
        # Convert context to JSON-serializable format
        state = {
            'metadata': self.context['metadata'],
            'config': self.context.get('config', {}),
            'artifacts_info': {
                key: str(type(value).__name__)
                for key, value in self.context.get('artifacts', {}).items()
            }
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.debug(f"Pipeline state saved to {state_file}")
    
    def execute(
        self,
        start_from: Optional[str] = None,
        stop_at: Optional[str] = None,
        skip_steps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute the pipeline.
        
        Args:
            start_from: Step name to start execution from (skip previous steps)
            stop_at: Step name to stop execution at (don't execute subsequent steps)
            skip_steps: List of step names to skip
        
        Returns:
            Final context dictionary
        """
        logger.info("=" * 60)
        logger.info(f"Executing Pipeline: {self.name}")
        logger.info("=" * 60)
        
        skip_steps = skip_steps or []
        start_idx = 0
        stop_idx = len(self.steps)
        
        # Find start index
        if start_from:
            for i, step in enumerate(self.steps):
                if step.name == start_from:
                    start_idx = i
                    logger.info(f"Starting from step: {start_from}")
                    break
        
        # Find stop index
        if stop_at:
            for i, step in enumerate(self.steps):
                if step.name == stop_at:
                    stop_idx = i + 1
                    logger.info(f"Stopping at step: {stop_at}")
                    break
        
        # Execute steps
        for i, step in enumerate(self.steps[start_idx:stop_idx], start=start_idx):
            if step.name in skip_steps:
                logger.info(f"Skipping step: {step.name}")
                continue
            
            if not step.enabled:
                logger.info(f"Step disabled: {step.name}")
                continue
            
            # Validate dependencies
            if not self._validate_dependencies(step):
                raise RuntimeError(
                    f"Dependencies not satisfied for step: {step.name}"
                )
            
            # Validate step prerequisites
            if not step.validate(self.context):
                logger.warning(
                    f"Validation failed for step: {step.name}. "
                    "Skipping execution."
                )
                continue
            
            # Execute step
            logger.info("-" * 60)
            logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            logger.info("-" * 60)
            
            try:
                self.context = step.execute(self.context)
                self.context['metadata']['steps_executed'].append(step.name)
                logger.info(f"✅ Step '{step.name}' completed successfully")
                
                # Save state after each step
                self._save_state()
                
            except Exception as e:
                logger.error(f"❌ Step '{step.name}' failed with error: {e}")
                raise
        
        logger.info("=" * 60)
        logger.info("Pipeline execution completed!")
        logger.info("=" * 60)
        
        return self.context
    
    def get_step(self, name: str) -> Optional[PipelineStep]:
        """
        Get a step by name.
        
        Args:
            name: Step name
        
        Returns:
            PipelineStep instance or None if not found
        """
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def enable_step(self, name: str):
        """Enable a step by name."""
        step = self.get_step(name)
        if step:
            step.enabled = True
            logger.info(f"Enabled step: {name}")
        else:
            logger.warning(f"Step not found: {name}")
    
    def disable_step(self, name: str):
        """Disable a step by name."""
        step = self.get_step(name)
        if step:
            step.enabled = False
            logger.info(f"Disabled step: {name}")
        else:
            logger.warning(f"Step not found: {name}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.
        
        Returns:
            Dictionary with pipeline status information
        """
        return {
            'name': self.name,
            'total_steps': len(self.steps),
            'executed_steps': len(self.context['metadata']['steps_executed']),
            'steps': [
                {
                    'name': step.name,
                    'enabled': step.enabled,
                    'executed': step.name in self.context['metadata']['steps_executed']
                }
                for step in self.steps
            ],
            'metadata': self.context['metadata']
        }

