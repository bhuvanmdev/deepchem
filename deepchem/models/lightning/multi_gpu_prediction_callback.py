"""
Multi-GPU Prediction Callback for PyTorch Lightning

This callback handles gathering predictions from all ranks in distributed (DDP) mode
to ensure that the complete dataset predictions are available on all ranks.
"""

import torch
import lightning as L
import numpy as np
from typing import Any, List, Optional
from lightning.pytorch.callbacks import Callback


class MultiGPUPredictionCallback(Callback):
    """
    A PyTorch Lightning callback that gathers predictions from all ranks in distributed mode.
    
    This callback ensures that when using multiple GPUs with DDP strategy, predictions
    from all devices are properly gathered and made available on all ranks.
    """
    
    def __init__(self):
        super().__init__()
        self.gathered_predictions: List[Any] = []
        self.all_predictions: List[Any] = []
        
    def on_predict_batch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """
        Called after each prediction batch. Stores predictions for later gathering.
        """
        if outputs is not None:
            self.all_predictions.append(outputs)
    
    def on_predict_epoch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule
    ) -> None:
        """
        Called at the end of prediction epoch. Gathers predictions from all ranks.
        """
        if not self.all_predictions:
            self.gathered_predictions = []
            return
            
        # Convert predictions to tensors for gathering
        gathered_results = []
        
        # Handle different prediction output formats
        if isinstance(self.all_predictions[0], (list, tuple)):
            # Multiple outputs per batch
            num_outputs = len(self.all_predictions[0])
            for output_idx in range(num_outputs):
                output_batches = [pred[output_idx] for pred in self.all_predictions]
                gathered_output = self._gather_predictions(output_batches, trainer)
                gathered_results.append(gathered_output)
        else:
            # Single output per batch
            gathered_output = self._gather_predictions(self.all_predictions, trainer)
            gathered_results = gathered_output
            
        self.gathered_predictions = gathered_results
    
    def _gather_predictions(self, predictions: List[Any], trainer: L.Trainer) -> Any:
        """
        Gather predictions from all ranks using all_gather.
        
        Parameters
        ----------
        predictions: List[Any]
            List of prediction arrays from this rank
        trainer: L.Trainer
            Lightning trainer instance
            
        Returns
        -------
        Any
            Gathered predictions from all ranks
        """
        if not predictions:
            return []
            
        # Concatenate predictions from this rank
        if isinstance(predictions[0], np.ndarray):
            local_preds = np.concatenate(predictions, axis=0)
        else:
            # Convert to numpy if needed
            local_preds = np.array(predictions)
            if local_preds.ndim > 1:
                local_preds = np.concatenate(local_preds, axis=0)
        
        # Convert to tensor for gathering
        local_tensor = torch.from_numpy(local_preds).to(trainer.strategy.root_device)
        
        # Gather from all ranks if we're in distributed mode
        if trainer.world_size > 1:
            try:
                # Use Lightning's all_gather utility
                gathered_tensors = trainer.strategy.all_gather(local_tensor)
                
                # Convert back to numpy and concatenate
                if isinstance(gathered_tensors, torch.Tensor):
                    # Single rank case or already concatenated
                    result = gathered_tensors.detach().cpu().numpy()
                elif isinstance(gathered_tensors, (list, tuple)):
                    # Multiple ranks - concatenate
                    all_arrays = [t.detach().cpu().numpy() for t in gathered_tensors]
                    result = np.concatenate(all_arrays, axis=0)
                else:
                    # Fallback to local predictions
                    result = local_preds
                    
            except Exception as e:
                # Fallback: return local predictions if gathering fails
                print(f"Warning: Failed to gather predictions from all ranks: {e}")
                result = local_preds
        else:
            # Single GPU - just return local predictions
            result = local_preds
            
        return result
    
    def on_predict_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset prediction storage at the start of prediction."""
        self.all_predictions = []
        self.gathered_predictions = []
