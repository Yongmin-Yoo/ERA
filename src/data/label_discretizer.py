"""
Label Discretizer Module.

Operationalizes patent renewal records as discrete value tiers
based on revealed economic preferences (Pakes, 1986).
"""

from typing import Optional


class LabelDiscretizer:
    """Converts continuous renewal counts into discrete value tiers.

    The tri-modal discretization is grounded in the marginal cost-benefit
    analysis of the EPO's progressive fee structure:
        - Class 1 (Low Value):    1-3 renewals  -> Early screening / fail-fast
        - Class 2 (Medium Value): 4-6 renewals  -> Standard commercialization cycle
        - Class 3 (High Value):   7+  renewals  -> Strategic long-term assets
    """

    TIER_MAP = {
        1: {"label": "Class 1 (Low Value)", "description": "Early Screening"},
        2: {"label": "Class 2 (Medium Value)", "description": "Standard Cycle"},
        3: {"label": "Class 3 (High Value)", "description": "Strategic Asset"},
    }

    def __init__(
        self,
        low_range: tuple = (1, 3),
        medium_range: tuple = (4, 6),
        high_range: tuple = (7, 20),
    ):
        self.low_range = low_range
        self.medium_range = medium_range
        self.high_range = high_range

    def discretize(self, renewal_count: int) -> Optional[int]:
        """Map a renewal count to a value tier (1, 2, or 3).

        Args:
            renewal_count: Number of times the patent was renewed.

        Returns:
            Integer class label (1, 2, or 3), or None if out of range.
        """
        if self.low_range[0] <= renewal_count <= self.low_range[1]:
            return 1
        elif self.medium_range[0] <= renewal_count <= self.medium_range[1]:
            return 2
        elif self.high_range[0] <= renewal_count <= self.high_range[1]:
            return 3
        return None

    def get_label_text(self, tier: int) -> str:
        """Return the human-readable label for a given tier."""
        return self.TIER_MAP[tier]["label"]

    def __repr__(self) -> str:
        return (
            f"LabelDiscretizer("
            f"low={self.low_range}, "
            f"medium={self.medium_range}, "
            f"high={self.high_range})"
        )
