import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

class TreeVisualizer:
    def __init__(self, tree, figsize=(40, 24)):
        self.tree = tree
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.node_positions = {}
        self.level_widths = defaultdict(float)
        
        # Calculate max depth to dynamically adjust parameters
        max_depth = self.get_tree_depth(tree)
        
        # Dynamic parameter adjustments
        self.level_height = 1.0 / (max_depth + 1)
        self.horizontal_spacing = min(0.1, 0.5 / max_depth)
        self.box_height = max(0.08, 0.12 / (max_depth ** 0.5))
        self.min_box_width = max(0.25, 0.35 / (max_depth ** 0.5))
        
    def get_tree_depth(self, node):
        """Calculate the maximum depth of the tree"""
        if node['type'] == 'leaf':
            return 0
        
        if 'children' not in node:
            return 0
        
        return 1 + max(self.get_tree_depth(child) for child in node['children'].values())

    def calculate_positions(self, node, level=0, x_offset=0):
        """Calculate positions for all nodes in the tree"""
        if node['type'] == 'leaf':
            width = self.min_box_width + self.horizontal_spacing
            self.level_widths[level] = max(self.level_widths[level], width)
            center = x_offset + width/2
            self.node_positions[id(node)] = (center, 1 - level * self.level_height)
            return width
        
        total_width = 0
        child_positions = []
        
        # Calculate total width needed for children
        if 'children' in node:
            children_widths = []
            for child in node['children'].values():
                child_width = self.calculate_positions(child, level + 1, x_offset + total_width)
                children_widths.append(child_width)
                total_width += child_width
                child_positions.append(x_offset + total_width - child_width/2)
            
            # Add extra spacing between children
            total_width += self.horizontal_spacing * (len(children_widths) - 1)
        
        # Ensure minimum width and center the node above its children
        width = max(total_width, self.min_box_width + self.horizontal_spacing)
        center = x_offset + width/2
        
        # Store position
        self.node_positions[id(node)] = (center, 1 - level * self.level_height)
        
        return width

    def draw_node(self, node, parent_pos=None):
        """Draw a node and its connections"""
        if id(node) not in self.node_positions:
            return
            
        x, y = self.node_positions[id(node)]
        
        # Dynamically adjust font sizes based on tree depth
        tree_depth = self.get_tree_depth(self.tree)
        font_size = max(6, 10 - tree_depth * 0.3)
        
        # Create node text and styling based on node type
        if node['type'] == 'split':
            feature_name = str(node['feature']).upper()
            samples = f"n={node['samples']:,}"  # Add thousands separator
            dist = [f"{k}:{v:,}" for k,v in node['distribution'].items()]
            dist_text = f"({', '.join(dist)})"
            confidence = f"Confidence: {node['confidence']:.2f}"
            
            if node.get('split_type') == 'continuous':
                split_info = f"\nSplit at {node['split_point']:.2f}"
            else:
                split_info = ""
                
            node_text = f"{feature_name}\n{samples}\n{dist_text}{split_info}\n{confidence}"
            box_color = '#E1F5FE'  # Lighter blue
            edge_color = '#0288D1'  # Darker blue for border
        else:
            pred = f"Prediction: {node['prediction']}"
            samples = f"n={node['samples']:,}"
            dist = [f"{k}:{v:,}" for k,v in node['distribution'].items()]
            dist_text = f"({', '.join(dist)})"
            probability = f"probability: {node['probability']:.2f}"
            
            node_text = f"{pred}\n{samples}\n{dist_text}\n{probability}"
            box_color = '#E8F5E9'  # Lighter green
            edge_color = '#388E3C'  # Darker green for border
        
        # Draw box with rounded corners
        box_width = self.min_box_width
        box = patches.FancyBboxPatch(
            (x - box_width/2, y - self.box_height/2),
            box_width, self.box_height,
            boxstyle=patches.BoxStyle("Round", pad=0.02),
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=1.5,
            alpha=0.9,
            zorder=1
        )
        self.ax.add_patch(box)
        
        # Add text with improved formatting
        self.ax.text(x, y, node_text,
                    ha='center',
                    va='center',
                    fontsize=font_size,
                    fontweight='bold',
                    linespacing=1.3,
                    zorder=2)
        
        # Draw connection to parent with improved styling
        if parent_pos is not None:
            # Calculate control points for curved line
            mid_y = (parent_pos[1] + y) / 2
            self.ax.plot([parent_pos[0], parent_pos[0], x, x],
                        [parent_pos[1], mid_y, mid_y, y],
                        color='#757575',
                        linestyle='-',
                        linewidth=1.5,
                        zorder=0)
            
            # Add edge label with better positioning and styling
            if 'edge_label' in node:
                mid_x = (parent_pos[0] + x) / 2
                # Position label slightly above the horizontal part of the connection
                label_y = mid_y + 0.02
                self.ax.text(mid_x, label_y,
                           str(node['edge_label']),
                           ha='center',
                           va='bottom',
                           fontsize=max(6, font_size - 2),
                           fontweight='bold',
                           bbox=dict(facecolor='white',
                                   edgecolor='none',
                                   alpha=0.7,
                                   pad=0.5))
        
        # Recursively draw children
        if node['type'] == 'split' and 'children' in node:
            for value, child in node['children'].items():
                child['edge_label'] = value
                self.draw_node(child, (x, y))

    def visualize(self):
        """Create the complete tree visualization"""
        # Calculate initial positions
        total_width = self.calculate_positions(self.tree)
        
        # Set plot limits with improved padding
        self.ax.set_xlim(-0.2, total_width + 0.2)
        self.ax.set_ylim(-0.2, 1.2)
        
        # Remove axes
        self.ax.axis('off')
        
        # Add title with improved styling
        self.ax.set_title('Cart Decision Tree Visualization: Parkison Disease Classification',
                         pad=20,
                         fontsize=16,
                         fontweight='bold')
        
        # Draw the tree
        self.draw_node(self.tree)
        
        # Add a subtle grid
        self.ax.grid(True, linestyle='--', alpha=0.1)
        
        # Adjust layout
        plt.tight_layout()
        
        return self.fig