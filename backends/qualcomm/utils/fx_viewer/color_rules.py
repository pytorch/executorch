"""Color rules for mapping node attributes to display colors."""

import hashlib
import colorsys

class ColorRule:
    """Base class for node->color mapping."""
    def __init__(self, attribute: str):
        self.attribute = attribute

    def apply(self, nodes_data: dict) -> tuple[dict, list]:
        """
        Takes a dictionary mapping node_id -> node_info_dict.
        Returns:
            - node_colors: Dict[str, str] mapping node_id -> hex color.
            - legend: List[Dict[str, str]] containing legend items {"label": ..., "color": ...}.
        """
        raise NotImplementedError

class CategoricalColorRule(ColorRule):
    """Assign deterministic colors to string/categorical values."""
    def __init__(self, attribute: str, color_map=None):
        super().__init__(attribute)
        self.color_map = color_map or {}

    def apply(self, nodes_data: dict) -> tuple[dict, list]:
        node_colors = {}
        unique_values = set()
        
        for node_id, data in nodes_data.items():
            if self.attribute not in data:
                continue
                
            val = data[self.attribute]
            if val is None:
                continue
                
            val_str = str(val)
            unique_values.add(val_str)
            
            if val_str in self.color_map:
                node_colors[node_id] = self.color_map[val_str]
            else:
                # Consistent hashing to a hue value in HSV space
                hash_val = int(hashlib.md5(val_str.encode('utf-8')).hexdigest(), 16)
                hue = (hash_val % 360) / 360.0 
                saturation = 0.65 
                value_hsv = 0.85
                
                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value_hsv)
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
                node_colors[node_id] = f"#{r:02x}{g:02x}{b:02x}"
                
        # Generate Legend
        legend = []
        # First add explicit map entries
        for k, v in self.color_map.items():
            if k in unique_values:
                legend.append({"label": str(k), "color": v})
                unique_values.remove(k)
        
        # Then add hashed ones
        for val_str in sorted(unique_values):
            # Recalculate hash for the legend to avoid storing it twice
            hash_val = int(hashlib.md5(val_str.encode('utf-8')).hexdigest(), 16)
            hue = (hash_val % 360) / 360.0 
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
            hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            legend.append({"label": str(val_str), "color": hex_color})
            
        return node_colors, legend

class NumericColorRule(ColorRule):
    """Assign gradient colors to numeric values."""
    def __init__(self, attribute: str, cmap="viridis", handle_outliers=True):
        super().__init__(attribute)
        self.cmap = cmap
        self.handle_outliers = handle_outliers

    def _interpolate_color(self, ratio):
        ratio = max(0.0, min(1.0, ratio))
        
        if self.cmap.lower() == 'reds':
            r = 255
            g = b = int(127 * (1 - ratio) + 128)
        elif self.cmap.lower() == 'blues':
            r = g = int(127 * (1 - ratio) + 128)
            b = 255
        elif self.cmap.lower() == 'greens':
            r = b = int(127 * (1 - ratio) + 128)
            g = 255
        else: # viridis-like fallback
            if ratio < 0.5:
                r, g, b = 68 + 50, int(1 + ratio * 2 * 120)+ 50, int(34 + ratio * 2 * 50) + 50
            else:
                r = int(68 + (ratio - 0.5) * 2 * 187)
                g = int(171 + (ratio - 0.5) * 2 * 84)
                b = int(134 - (ratio - 0.5) * 2 * 134)
                
        return f"#{r:02x}{g:02x}{b:02x}"

    def apply(self, nodes_data: dict) -> tuple[dict, list]:
        # Pass 1: Collect valid values for fitting
        valid_values = []
        for data in nodes_data.values():
            if self.attribute in data:
                val = data[self.attribute]
                if isinstance(val, (int, float)):
                    valid_values.append(val)
                    
        if not valid_values:
            return {}, []
            
        # Fit bounds
        if self.handle_outliers and len(valid_values) > 10:
            valid_values.sort()
            p5_idx = max(0, int(len(valid_values) * 0.05))
            p95_idx = min(len(valid_values) - 1, int(len(valid_values) * 0.95))
            _min = valid_values[p5_idx]
            _max = valid_values[p95_idx]
        else:
            _min = min(valid_values)
            _max = max(valid_values)
            
        if _min == _max:
            _max = _min + 1e-9

        # Pass 2: Calculate colors
        node_colors = {}
        for node_id, data in nodes_data.items():
            if self.attribute in data:
                val = data[self.attribute]
                if isinstance(val, (int, float)):
                    ratio = (val - _min) / (_max - _min)
                    node_colors[node_id] = self._interpolate_color(ratio)
                    
        # Generate Legend
        legend = []
        for i in range(5):
            ratio = i / 4.0
            val = _min + ratio * (_max - _min)
            color = self._interpolate_color(ratio)
            
            if abs(val) >= 1000 or (abs(val) < 0.01 and val != 0):
                label_str = f"{val:.2e}"
            elif isinstance(val, int) or float(val).is_integer():
                label_str = f"{int(val)}"
            else:
                label_str = f"{val:.2f}"
            legend.append({"label": label_str, "color": color})
            
        return node_colors, legend
