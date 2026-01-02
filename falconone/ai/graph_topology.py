"""
FalconOne Graph Neural Network Topology Inference Module (v1.5.4)
Uses GNN to reconstruct cellular network topology and predict handovers
Author: FalconOne Development Team
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Create dummy layers class for fallback
    class DummyLayers:
        class Layer:
            def __init__(self, *args, **kwargs):
                pass
    layers = DummyLayers()
    print("[WARNING] TensorFlow not installed. GNN topology inference disabled.")

try:
    from ..utils.logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent is None else parent.getChild(name)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")
        def debug(self, msg, **kw): self.logger.debug(f"{msg} {kw if kw else ''}")


class CellNode:
    """Represents a cellular base station node"""
    
    def __init__(self, cell_id: int, cell_type: str, frequency_mhz: float,
                 location: Tuple[float, float] = None):
        self.cell_id = cell_id
        self.cell_type = cell_type  # GSM, LTE, 5G, etc.
        self.frequency_mhz = frequency_mhz
        self.location = location  # (latitude, longitude)
        self.neighbors = []  # Adjacent cells (handover candidates)
        self.features = {}  # Node features (RSRP, load, etc.)
        
    def add_neighbor(self, neighbor_id: int, handover_count: int = 1):
        """Add neighbor cell with handover frequency"""
        self.neighbors.append({'cell_id': neighbor_id, 'handover_count': handover_count})
    
    def get_feature_vector(self) -> np.ndarray:
        """Extract node features as vector"""
        return np.array([
            self.cell_id,
            self._encode_cell_type(),
            self.frequency_mhz,
            len(self.neighbors),
            self.features.get('avg_rsrp', -100),
            self.features.get('cell_load', 0.5)
        ])
    
    def _encode_cell_type(self) -> int:
        """Encode cell type as integer"""
        type_map = {'GSM': 0, 'UMTS': 1, 'CDMA': 2, 'LTE': 3, '5G': 4, '6G': 5}
        return type_map.get(self.cell_type, -1)


class NetworkTopology:
    """Network topology graph structure"""
    
    def __init__(self):
        self.cells: Dict[int, CellNode] = {}
        self.handover_events: List[Tuple[int, int]] = []  # (source_cell, target_cell)
        
    def add_cell(self, cell: CellNode):
        """Add cell to topology"""
        self.cells[cell.cell_id] = cell
    
    def record_handover(self, source_cell_id: int, target_cell_id: int):
        """Record handover event"""
        self.handover_events.append((source_cell_id, target_cell_id))
        
        # Update neighbor relationships
        if source_cell_id in self.cells:
            found = False
            for neighbor in self.cells[source_cell_id].neighbors:
                if neighbor['cell_id'] == target_cell_id:
                    neighbor['handover_count'] += 1
                    found = True
                    break
            if not found:
                self.cells[source_cell_id].add_neighbor(target_cell_id)
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get graph adjacency matrix"""
        n_cells = len(self.cells)
        adj_matrix = np.zeros((n_cells, n_cells))
        
        cell_ids = sorted(self.cells.keys())
        id_to_idx = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
        
        for cell_id, cell in self.cells.items():
            for neighbor in cell.neighbors:
                src_idx = id_to_idx[cell_id]
                tgt_idx = id_to_idx.get(neighbor['cell_id'])
                if tgt_idx is not None:
                    adj_matrix[src_idx][tgt_idx] = neighbor['handover_count']
        
        return adj_matrix
    
    def get_node_features(self) -> np.ndarray:
        """Get node feature matrix"""
        cell_ids = sorted(self.cells.keys())
        features = []
        for cell_id in cell_ids:
            features.append(self.cells[cell_id].get_feature_vector())
        return np.array(features)


class GraphConvolutionLayer(layers.Layer):
    """Graph Convolutional Network layer"""
    
    def __init__(self, output_dim, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[1][-1], self.output_dim),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.output_dim,),
            initializer='zeros',
            name='bias'
        )
        super().build(input_shape)
    
    def call(self, inputs):
        """Forward pass: [adjacency_matrix, node_features]"""
        adj, features = inputs
        
        # Normalize adjacency matrix
        degree = tf.reduce_sum(adj, axis=1)
        degree = tf.maximum(degree, 1.0)  # Avoid division by zero
        degree_inv = tf.pow(degree, -0.5)
        degree_mat = tf.linalg.diag(degree_inv)
        adj_normalized = tf.matmul(tf.matmul(degree_mat, adj), degree_mat)
        
        # Graph convolution: A_norm @ X @ W
        output = tf.matmul(features, self.kernel)
        output = tf.matmul(adj_normalized, output)
        output = tf.nn.bias_add(output, self.bias)
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output


class GNNTopologyInference:
    """
    Graph Neural Network for cellular network topology inference
    Predicts handover patterns and UE movement trajectories
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        self.config = config
        
        # Handle logger initialization
        if logger is None:
            logger = logging.getLogger('FalconOne')
        self.logger = ModuleLogger('AI-GNNTopology', logger)
        
        # Network topology
        self.topology = NetworkTopology()
        
        # GNN model
        self.model = None
        self._model_built = False
        
        # Configuration
        self.embedding_dim = config.get('ai_ml.gnn.embedding_dim', 64) if hasattr(config, 'get') else 64
        self.num_gnn_layers = config.get('ai_ml.gnn.num_layers', 3) if hasattr(config, 'get') else 3
        
        if TF_AVAILABLE:
            self.logger.info("GNN Topology Inference initialized (v1.5.4)",
                           embedding_dim=self.embedding_dim,
                           num_layers=self.num_gnn_layers)
        else:
            self.logger.warning("TensorFlow not available - GNN disabled")
    
    def build_model(self):
        """Build GNN model for handover prediction"""
        if not TF_AVAILABLE:
            self.logger.error("Cannot build GNN - TensorFlow not available")
            return
        
        try:
            # Input: adjacency matrix and node features
            adj_input = layers.Input(shape=(None, None), name='adjacency_matrix')
            feature_input = layers.Input(shape=(None, 6), name='node_features')  # 6 features per node
            
            # GNN layers
            x = feature_input
            for i in range(self.num_gnn_layers):
                x = GraphConvolutionLayer(
                    self.embedding_dim,
                    activation='relu',
                    name=f'gcn_{i}'
                )([adj_input, x])
            
            # Readout layer for graph-level prediction
            graph_embedding = layers.GlobalAveragePooling1D()(x)
            
            # Predict handover probability matrix
            handover_logits = layers.Dense(
                None,  # Dynamic based on number of cells
                activation='softmax',
                name='handover_predictions'
            )(graph_embedding)
            
            # Node embedding output (for clustering/visualization)
            node_embeddings = layers.Dense(32, activation='relu', name='node_embeddings')(x)
            
            self.model = keras.Model(
                inputs=[adj_input, feature_input],
                outputs=[handover_logits, node_embeddings]
            )
            
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss={
                    'handover_predictions': 'categorical_crossentropy',
                    'node_embeddings': None  # Unsupervised embedding
                },
                metrics={
                    'handover_predictions': ['accuracy']
                }
            )
            
            self._model_built = True
            self.logger.info("GNN model built successfully",
                           params=self.model.count_params() if hasattr(self.model, 'count_params') else 'N/A')
            
        except Exception as e:
            self.logger.error(f"GNN model build failed: {e}")
    
    def add_cell(self, cell_id: int, cell_type: str, frequency_mhz: float,
                location: Tuple[float, float] = None, features: Dict[str, Any] = None):
        """Add cell to topology"""
        cell = CellNode(cell_id, cell_type, frequency_mhz, location)
        if features:
            cell.features = features
        self.topology.add_cell(cell)
        self.logger.debug(f"Added cell {cell_id} ({cell_type}) to topology")
    
    def record_handover(self, source_cell_id: int, target_cell_id: int):
        """Record handover event to build topology"""
        self.topology.record_handover(source_cell_id, target_cell_id)
        self.logger.debug(f"Recorded handover: {source_cell_id} -> {target_cell_id}")
    
    def infer_topology(self) -> Dict[str, Any]:
        """
        Infer network topology from observed handovers
        
        Returns:
            Inferred topology metadata
        """
        if not self.topology.cells:
            return {'error': 'no_cells_in_topology'}
        
        adj_matrix = self.topology.get_adjacency_matrix()
        node_features = self.topology.get_node_features()
        
        self.logger.info("Inferred network topology",
                       num_cells=len(self.topology.cells),
                       num_handovers=len(self.topology.handover_events))
        
        # Identify hub cells (high handover frequency)
        handover_counts = np.sum(adj_matrix, axis=1)
        hub_cells = np.argsort(handover_counts)[-5:]  # Top 5 hubs
        
        cell_ids = sorted(self.topology.cells.keys())
        
        return {
            'num_cells': len(self.topology.cells),
            'num_handover_events': len(self.topology.handover_events),
            'hub_cells': [cell_ids[idx] for idx in hub_cells],
            'adjacency_matrix': adj_matrix.tolist(),
            'avg_neighbors_per_cell': float(np.mean([len(c.neighbors) for c in self.topology.cells.values()]))
        }
    
    def predict_handover(self, source_cell_id: int, ue_features: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predict next handover target for UE
        
        Args:
            source_cell_id: Current serving cell
            ue_features: Optional UE-specific features (RSRP, speed, direction)
            
        Returns:
            Predicted target cells with probabilities
        """
        if not self._model_built:
            self.logger.warning("GNN model not built, using heuristic prediction")
            return self._heuristic_predict_handover(source_cell_id)
        
        if source_cell_id not in self.topology.cells:
            return {'error': 'cell_not_in_topology'}
        
        try:
            adj_matrix = self.topology.get_adjacency_matrix()
            node_features = self.topology.get_node_features()
            
            # Predict using GNN
            adj_tensor = tf.constant(adj_matrix[np.newaxis, :, :], dtype=tf.float32)
            features_tensor = tf.constant(node_features[np.newaxis, :, :], dtype=tf.float32)
            
            handover_probs, _ = self.model.predict([adj_tensor, features_tensor], verbose=0)
            
            # Extract probabilities for source cell
            cell_ids = sorted(self.topology.cells.keys())
            source_idx = cell_ids.index(source_cell_id)
            
            # Get top 3 predictions
            top_indices = np.argsort(handover_probs[source_idx])[-3:][::-1]
            predictions = [
                {
                    'cell_id': cell_ids[idx],
                    'probability': float(handover_probs[source_idx][idx])
                }
                for idx in top_indices
            ]
            
            self.logger.info(f"Predicted handover from cell {source_cell_id}",
                           top_target=predictions[0]['cell_id'])
            
            return {
                'source_cell_id': source_cell_id,
                'predictions': predictions,
                'model': 'gnn'
            }
            
        except Exception as e:
            self.logger.error(f"Handover prediction failed: {e}")
            return {'error': str(e)}
    
    def _heuristic_predict_handover(self, source_cell_id: int) -> Dict[str, Any]:
        """Heuristic handover prediction (fallback when GNN unavailable)"""
        if source_cell_id not in self.topology.cells:
            return {'error': 'cell_not_in_topology'}
        
        neighbors = self.topology.cells[source_cell_id].neighbors
        if not neighbors:
            return {'source_cell_id': source_cell_id, 'predictions': [], 'model': 'heuristic'}
        
        # Sort by handover count
        sorted_neighbors = sorted(neighbors, key=lambda x: x['handover_count'], reverse=True)
        
        predictions = [
            {
                'cell_id': n['cell_id'],
                'probability': n['handover_count'] / sum(nb['handover_count'] for nb in neighbors)
            }
            for n in sorted_neighbors[:3]
        ]
        
        return {
            'source_cell_id': source_cell_id,
            'predictions': predictions,
            'model': 'heuristic'
        }
    
    def identify_rogue_cell_opportunity(self) -> List[Dict[str, Any]]:
        """
        Identify optimal locations for rogue cell deployment
        Based on handover patterns and coverage gaps
        
        Returns:
            List of recommended rogue cell positions
        """
        recommendations = []
        
        # Find cells with high handover frequency (handover bottlenecks)
        for cell_id, cell in self.topology.cells.items():
            total_handovers = sum(n['handover_count'] for n in cell.neighbors)
            if total_handovers > 10:  # Threshold for "hot" cell
                recommendations.append({
                    'target_cell_id': cell_id,
                    'cell_type': cell.cell_type,
                    'handover_frequency': total_handovers,
                    'location': cell.location,
                    'reason': 'high_handover_frequency'
                })
        
        self.logger.info(f"Identified {len(recommendations)} rogue cell opportunities")
        
        return recommendations
    
    def visualize_topology(self) -> Dict[str, Any]:
        """
        Generate topology visualization data
        
        Returns:
            Graph data for visualization (nodes, edges, positions)
        """
        nodes = []
        edges = []
        
        for cell_id, cell in self.topology.cells.items():
            nodes.append({
                'id': cell_id,
                'type': cell.cell_type,
                'frequency': cell.frequency_mhz,
                'location': cell.location
            })
            
            for neighbor in cell.neighbors:
                edges.append({
                    'source': cell_id,
                    'target': neighbor['cell_id'],
                    'weight': neighbor['handover_count']
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }
    
    def train_model(self, epochs: int = 50):
        """
        Train GNN model on observed topology
        
        Args:
            epochs: Number of training epochs
        """
        if not TF_AVAILABLE or not self._model_built:
            self.logger.error("Cannot train - GNN model not available")
            return
        
        try:
            adj_matrix = self.topology.get_adjacency_matrix()
            node_features = self.topology.get_node_features()
            
            # Generate training labels (next handover targets)
            # Simplified: Use observed handover frequency as soft labels
            handover_labels = adj_matrix / (np.sum(adj_matrix, axis=1, keepdims=True) + 1e-9)
            
            # Expand dimensions for batch
            adj_tensor = adj_matrix[np.newaxis, :, :]
            features_tensor = node_features[np.newaxis, :, :]
            labels_tensor = handover_labels[np.newaxis, :]
            
            self.logger.info(f"Training GNN model for {epochs} epochs...")
            
            history = self.model.fit(
                [adj_tensor, features_tensor],
                {'handover_predictions': labels_tensor},
                epochs=epochs,
                verbose=0
            )
            
            final_loss = history.history['loss'][-1]
            self.logger.info(f"GNN training completed", final_loss=f"{final_loss:.4f}")
            
            return history
            
        except Exception as e:
            self.logger.error(f"GNN training failed: {e}")
            return None
    
    def get_topology_summary(self) -> Dict[str, Any]:
        """Get comprehensive topology summary"""
        return {
            'num_cells': len(self.topology.cells),
            'num_handover_events': len(self.topology.handover_events),
            'cell_types': list(set(c.cell_type for c in self.topology.cells.values())),
            'model_built': self._model_built,
            'timestamp': datetime.now().isoformat()
        }
