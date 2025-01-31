from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from rdflib import Graph
import yaml
import datetime

@dataclass
class ShaclShape:
    shape_id: str
    legal_text: str
    graph: Graph
    created_at: datetime.datetime
    updated_at: datetime.datetime
    description: Optional[str] = None

class ShapeStore:
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.shapes: Dict[str, ShaclShape] = {}
        self.load_all_shapes()
    
    def add_shape(self, shape_id: str, legal_text: str, graph: Graph, description: Optional[str] = None) -> ShaclShape:
        """Add a new SHACL shape."""
        now = datetime.datetime.now()
        shape = ShaclShape(
            shape_id=shape_id,
            legal_text=legal_text,
            graph=graph,
            created_at=now,
            updated_at=now,
            description=description
        )
        self.shapes[shape_id] = shape
        self._save_shape(shape)
        return shape
    
    def update_shape(self, shape_id: str, graph: Graph, description: Optional[str] = None) -> ShaclShape:
        """Update an existing shape."""
        if shape_id not in self.shapes:
            raise ValueError(f"Shape not found: {shape_id}")
        
        shape = self.shapes[shape_id]
        shape.graph = graph
        shape.updated_at = datetime.datetime.now()
        if description is not None:
            shape.description = description
        
        self._save_shape(shape)
        return shape
    
    def get_shape(self, shape_id: str) -> Optional[ShaclShape]:
        """Get a shape by ID."""
        return self.shapes.get(shape_id)
    
    def delete_shape(self, shape_id: str) -> None:
        """Delete a shape."""
        if shape_id not in self.shapes:
            raise ValueError(f"Shape not found: {shape_id}")
        
        shape_dir = self.store_dir / shape_id
        if shape_dir.exists():
            for file in shape_dir.iterdir():
                file.unlink()
            shape_dir.rmdir()
        
        del self.shapes[shape_id]
    
    def _save_shape(self, shape: ShaclShape) -> None:
        """Save shape to disk."""
        shape_dir = self.store_dir / shape.shape_id
        shape_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            'shape_id': shape.shape_id,
            'created_at': shape.created_at.isoformat(),
            'updated_at': shape.updated_at.isoformat(),
            'description': shape.description
        }
        with open(shape_dir / 'metadata.yaml', 'w') as f:
            yaml.dump(metadata, f)
        
        # Save legal text
        with open(shape_dir / 'legal_text.txt', 'w') as f:
            f.write(shape.legal_text)
        
        # Save SHACL graph
        shape.graph.serialize(destination=str(shape_dir / 'shape.ttl'), format='turtle')
    
    def load_all_shapes(self) -> None:
        """Load all shapes from disk."""
        for shape_dir in self.store_dir.iterdir():
            if not shape_dir.is_dir():
                continue
            
            try:
                # Load metadata
                with open(shape_dir / 'metadata.yaml', 'r') as f:
                    metadata = yaml.safe_load(f)
                
                # Load legal text
                with open(shape_dir / 'legal_text.txt', 'r') as f:
                    legal_text = f.read()
                
                # Load graph
                g = Graph()
                g.parse(shape_dir / 'shape.ttl', format='turtle')
                
                shape = ShaclShape(
                    shape_id=metadata['shape_id'],
                    legal_text=legal_text,
                    graph=g,
                    created_at=datetime.datetime.fromisoformat(metadata['created_at']),
                    updated_at=datetime.datetime.fromisoformat(metadata['updated_at']),
                    description=metadata.get('description')
                )
                self.shapes[shape.shape_id] = shape
            except Exception as e:
                print(f"Error loading shape from {shape_dir}: {e}") 