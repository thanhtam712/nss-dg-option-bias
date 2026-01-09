"""
NSS Controller - Intelligent NSS application control
Decides when to apply NSS direction based on various criteria
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple


class NSSController:
    """
    Controller to decide when NSS direction should be applied during inference.
    
    Strategies:
    1. Always: Always apply NSS
    2. Confidence-based: Apply only when model confidence is low
    3. Uncertainty-based: Apply when model is uncertain
    4. Adaptive: Dynamically adjust alpha based on context
    5. Selective: Apply only for specific question types
    """
    
    def __init__(
        self,
        strategy: str = "always",
        confidence_threshold: float = 0.7,
        uncertainty_threshold: float = 0.3,
        alpha_image_base: float = 0.3,
        alpha_text_base: float = 0.3,
        enable_dynamic_alpha: bool = True,
        verbose: bool = True
    ):
        """
        Initialize NSS Controller.
        
        Args:
            strategy: "always", "confidence", "uncertainty", "adaptive", "selective"
            confidence_threshold: Threshold for confidence-based strategy
            uncertainty_threshold: Threshold for uncertainty-based strategy
            alpha_image_base: Base alpha for visual NSS
            alpha_text_base: Base alpha for textual NSS
            enable_dynamic_alpha: Whether to dynamically adjust alpha
            verbose: Print decision logs
        """
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.alpha_image_base = alpha_image_base
        self.alpha_text_base = alpha_text_base
        self.enable_dynamic_alpha = enable_dynamic_alpha
        self.verbose = verbose
        
        # Statistics tracking
        self.stats = {
            "total_queries": 0,
            "nss_applied": 0,
            "nss_skipped": 0,
            "avg_confidence_when_applied": [],
            "avg_confidence_when_skipped": []
        }
    
    def should_apply_nss(
        self,
        logits: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        question_text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, float, float, Dict[str, Any]]:
        """
        Decide whether to apply NSS direction.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            hidden_states: Model hidden states
            question_text: Question text for selective strategy
            context: Additional context (image count, etc.)
        
        Returns:
            should_apply: Whether to apply NSS
            alpha_image: Adjusted alpha for visual NSS
            alpha_text: Adjusted alpha for textual NSS
            info: Dictionary with decision information
        """
        self.stats["total_queries"] += 1
        
        info = {
            "strategy": self.strategy,
            "reason": "",
            "confidence": None,
            "uncertainty": None,
            "alpha_adjustment": 1.0
        }
        
        # Strategy: Always
        if self.strategy == "always":
            info["reason"] = "Always apply NSS"
            self.stats["nss_applied"] += 1
            return True, self.alpha_image_base, self.alpha_text_base, info
        
        # Strategy: Confidence-based
        if self.strategy == "confidence":
            if logits is None:
                if self.verbose:
                    print("⚠️  Logits not provided, falling back to always apply")
                return True, self.alpha_image_base, self.alpha_text_base, info
            
            confidence = self._compute_confidence(logits)
            info["confidence"] = confidence
            
            if confidence < self.confidence_threshold:
                # Low confidence → Apply NSS
                info["reason"] = f"Low confidence ({confidence:.3f} < {self.confidence_threshold})"
                self.stats["nss_applied"] += 1
                self.stats["avg_confidence_when_applied"].append(confidence)
                
                # Dynamic alpha adjustment
                if self.enable_dynamic_alpha:
                    # Lower confidence → Higher alpha
                    alpha_boost = 1.0 + (1.0 - confidence / self.confidence_threshold)
                    alpha_image = min(self.alpha_image_base * alpha_boost, 1.0)
                    alpha_text = min(self.alpha_text_base * alpha_boost, 1.0)
                    info["alpha_adjustment"] = alpha_boost
                    return True, alpha_image, alpha_text, info
                
                return True, self.alpha_image_base, self.alpha_text_base, info
            else:
                # High confidence → Skip NSS
                info["reason"] = f"High confidence ({confidence:.3f} >= {self.confidence_threshold})"
                self.stats["nss_skipped"] += 1
                self.stats["avg_confidence_when_skipped"].append(confidence)
                return False, 0.0, 0.0, info
        
        # Strategy: Uncertainty-based
        if self.strategy == "uncertainty":
            if logits is None:
                if self.verbose:
                    print("⚠️  Logits not provided, falling back to always apply")
                return True, self.alpha_image_base, self.alpha_text_base, info
            
            uncertainty = self._compute_uncertainty(logits)
            info["uncertainty"] = uncertainty
            
            if uncertainty > self.uncertainty_threshold:
                # High uncertainty → Apply NSS
                info["reason"] = f"High uncertainty ({uncertainty:.3f} > {self.uncertainty_threshold})"
                self.stats["nss_applied"] += 1
                
                # Dynamic alpha adjustment
                if self.enable_dynamic_alpha:
                    # Higher uncertainty → Higher alpha
                    alpha_boost = 1.0 + (uncertainty / self.uncertainty_threshold - 1.0)
                    alpha_image = min(self.alpha_image_base * alpha_boost, 1.0)
                    alpha_text = min(self.alpha_text_base * alpha_boost, 1.0)
                    info["alpha_adjustment"] = alpha_boost
                    return True, alpha_image, alpha_text, info
                
                return True, self.alpha_image_base, self.alpha_text_base, info
            else:
                # Low uncertainty → Skip NSS
                info["reason"] = f"Low uncertainty ({uncertainty:.3f} <= {self.uncertainty_threshold})"
                self.stats["nss_skipped"] += 1
                return False, 0.0, 0.0, info
        
        # Strategy: Adaptive
        if self.strategy == "adaptive":
            # Combine multiple signals
            signals = []
            
            if logits is not None:
                confidence = self._compute_confidence(logits)
                uncertainty = self._compute_uncertainty(logits)
                info["confidence"] = confidence
                info["uncertainty"] = uncertainty
                
                # Low confidence or high uncertainty → Apply NSS
                if confidence < self.confidence_threshold or uncertainty > self.uncertainty_threshold:
                    signals.append("confidence_uncertainty")
            
            # Check question type
            if question_text is not None:
                if self._is_hallucination_prone(question_text):
                    signals.append("hallucination_prone_question")
            
            # Check context
            if context is not None:
                if context.get("num_images", 0) > 10:
                    signals.append("many_images")
                if context.get("video_length", 0) > 60:
                    signals.append("long_video")
            
            if len(signals) > 0:
                info["reason"] = f"Adaptive signals: {', '.join(signals)}"
                self.stats["nss_applied"] += 1
                
                # Dynamic alpha based on number of signals
                if self.enable_dynamic_alpha:
                    alpha_boost = 1.0 + 0.2 * len(signals)
                    alpha_image = min(self.alpha_image_base * alpha_boost, 1.0)
                    alpha_text = min(self.alpha_text_base * alpha_boost, 1.0)
                    info["alpha_adjustment"] = alpha_boost
                    return True, alpha_image, alpha_text, info
                
                return True, self.alpha_image_base, self.alpha_text_base, info
            else:
                info["reason"] = "No adaptive signals detected"
                self.stats["nss_skipped"] += 1
                return False, 0.0, 0.0, info
        
        # Strategy: Selective (question-type based)
        if self.strategy == "selective":
            if question_text is None:
                if self.verbose:
                    print("⚠️  Question text not provided, falling back to always apply")
                return True, self.alpha_image_base, self.alpha_text_base, info
            
            if self._is_hallucination_prone(question_text):
                info["reason"] = "Hallucination-prone question type"
                self.stats["nss_applied"] += 1
                return True, self.alpha_image_base, self.alpha_text_base, info
            else:
                info["reason"] = "Simple question type"
                self.stats["nss_skipped"] += 1
                return False, 0.0, 0.0, info
        
        # Default: Apply NSS
        info["reason"] = "Default behavior"
        self.stats["nss_applied"] += 1
        return True, self.alpha_image_base, self.alpha_text_base, info
    
    def _compute_confidence(self, logits: torch.Tensor) -> float:
        """
        Compute model confidence from logits.
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
        
        Returns:
            confidence: Normalized confidence score [0, 1]
        """
        # Get the last token's logits
        if logits.dim() == 3:
            last_logits = logits[0, -1, :]  # [vocab_size]
        elif logits.dim() == 2:
            last_logits = logits[-1, :]  # [vocab_size]
        else:
            last_logits = logits
        
        # Compute softmax probabilities
        probs = torch.softmax(last_logits, dim=-1)
        
        # Confidence = max probability
        confidence = probs.max().item()
        
        return confidence
    
    def _compute_uncertainty(self, logits: torch.Tensor) -> float:
        """
        Compute model uncertainty from logits using entropy.
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
        
        Returns:
            uncertainty: Normalized entropy [0, 1]
        """
        # Get the last token's logits
        if logits.dim() == 3:
            last_logits = logits[0, -1, :]  # [vocab_size]
        elif logits.dim() == 2:
            last_logits = logits[-1, :]  # [vocab_size]
        else:
            last_logits = logits
        
        # Compute softmax probabilities
        probs = torch.softmax(last_logits, dim=-1)
        
        # Compute entropy: H = -Σ p(x) log p(x)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        # Normalize by max entropy (log of vocab size)
        vocab_size = last_logits.shape[-1]
        max_entropy = np.log(vocab_size)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def _is_hallucination_prone(self, question_text: str) -> bool:
        """
        Check if question is prone to hallucination.
        
        Hallucination-prone questions typically:
        - Ask about specific details
        - Require counting
        - Ask about relationships/positions
        - Require temporal reasoning
        
        Args:
            question_text: Question text
        
        Returns:
            is_prone: Whether question is hallucination-prone
        """
        question_lower = question_text.lower()
        
        # Keywords indicating hallucination-prone questions
        prone_keywords = [
            # Counting
            "how many", "count", "number of",
            # Spatial relationships
            "where", "position", "location", "left", "right", "above", "below",
            # Temporal reasoning
            "when", "before", "after", "first", "last", "sequence",
            # Fine-grained details
            "color of", "what color", "specific", "exactly",
            # Object relationships
            "between", "next to", "near", "far from",
            # Attributes
            "size", "shape", "texture", "material"
        ]
        
        for keyword in prone_keywords:
            if keyword in question_lower:
                return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about NSS application decisions.
        
        Returns:
            stats: Dictionary with statistics
        """
        stats = self.stats.copy()
        
        if stats["total_queries"] > 0:
            stats["nss_application_rate"] = stats["nss_applied"] / stats["total_queries"]
        
        if len(stats["avg_confidence_when_applied"]) > 0:
            stats["avg_confidence_when_applied"] = np.mean(stats["avg_confidence_when_applied"])
        else:
            stats["avg_confidence_when_applied"] = None
        
        if len(stats["avg_confidence_when_skipped"]) > 0:
            stats["avg_confidence_when_skipped"] = np.mean(stats["avg_confidence_when_skipped"])
        else:
            stats["avg_confidence_when_skipped"] = None
        
        return stats
    
    def print_statistics(self):
        """Print statistics in a formatted way."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("NSS Controller Statistics")
        print("="*60)
        print(f"Strategy: {self.strategy}")
        print(f"Total queries: {stats['total_queries']}")
        print(f"NSS applied: {stats['nss_applied']} ({stats.get('nss_application_rate', 0)*100:.1f}%)")
        print(f"NSS skipped: {stats['nss_skipped']} ({(1-stats.get('nss_application_rate', 0))*100:.1f}%)")
        
        if stats["avg_confidence_when_applied"] is not None:
            print(f"Avg confidence (applied): {stats['avg_confidence_when_applied']:.3f}")
        
        if stats["avg_confidence_when_skipped"] is not None:
            print(f"Avg confidence (skipped): {stats['avg_confidence_when_skipped']:.3f}")
        
        print("="*60 + "\n")


def create_nss_controller(args) -> Optional[NSSController]:
    """
    Create NSS controller from command line arguments.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        controller: NSSController instance or None if NSS disabled
    """
    # Check if NSS is enabled
    if args.alpha_image == 0 and args.alpha_text == 0:
        return None
    
    strategy = getattr(args, 'nss_strategy', 'always')
    
    controller = NSSController(
        strategy=strategy,
        confidence_threshold=getattr(args, 'nss_confidence_threshold', 0.7),
        uncertainty_threshold=getattr(args, 'nss_uncertainty_threshold', 0.3),
        alpha_image_base=args.alpha_image,
        alpha_text_base=args.alpha_text,
        enable_dynamic_alpha=getattr(args, 'nss_enable_dynamic_alpha', True),
        verbose=getattr(args, 'nss_verbose', True)
    )
    
    return controller
