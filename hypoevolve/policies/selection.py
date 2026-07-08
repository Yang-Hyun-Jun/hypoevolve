"""Selection policies for archive-driven parent sampling."""

from __future__ import annotations

import random
from typing import Any, Optional

from hypoevolve.memory.coulomb_archive import ArchiveEntry


class CoulombSelectionPolicy:
    """Delegate parent selection to a Coulomb archive's own repulsive sampler."""

    def select(
        self,
        archive: Any,
        rng: Optional[random.Random] = None,
    ) -> ArchiveEntry:
        """Sample one parent by delegating to the archive itself.

        Args:
            archive: The Coulomb archive to sample from.
            rng: Optional random number generator.

        Returns:
            ArchiveEntry: The selected parent entry.
        """
        return archive.sample_parent(rng)
