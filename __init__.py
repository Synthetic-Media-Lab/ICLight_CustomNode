"""
@title: FAL.ai API Nodes for ComfyUI
@nickname: FAL.ai Nodes
@description: Nodes for interacting with FAL.ai API services, including ICLight v2 for image relighting.
"""

# Import directly from the implementation file
from .fal_iclight_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# This is what ComfyUI looks for when loading nodes
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']