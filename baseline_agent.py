# baseline_agent.py

import numpy as np

class BaselineAgent:
    """
    Bandwidth-weighted random relay selection.

    Implements a simplified version of Tor's current relay selection algorithm
    where relays are chosen with probability proportional to their bandwidth.
    
    Reference: https://spec.torproject.org/path-spec/
    """

    def select_action(self, action_mask, relays):
        """
        Select a relay using bandwidth-weighted random sampling.

        Args:
            action_mask: Boolean array indicating valid relay choices
            relays: List of relay dictionaries
        
        Returns:
            int: Index of the selected relay
        """

        # List of relay IDs that are valid for selection
        valid_actions = np.where(action_mask)[0]

        # Extract bandwidths for valid relays
        bandwidths = np.array([relays[i]['bandwidth'] for i in valid_actions])
        
        # Normalize bandwidths to probability distribution (sums to 1.0)
        probabilities = bandwidths / bandwidths.sum()

        # Randomly select relay weighted by bandwidth
        selected_action = np.random.choice(valid_actions, p=probabilities)

        return selected_action