from dataclasses import dataclass
from pathlib import Path
import yaml
from rdflib import Graph
from typing import List, Dict, Optional

@dataclass
class ExampleMapping:
    legal_text: str
    shacl_shape: Graph
    annotations: Optional[Dict[str, str]] = None  # Store any manual annotations/explanations
    
class ExampleStore:
    def __init__(self, examples_dir: Path):
        self.examples_dir = examples_dir
        self.examples: List[ExampleMapping] = []
        
    def add_example(self, legal_text_path: Path, shacl_shape_path: Path, 
                   annotations_path: Optional[Path] = None) -> None:
        """Add a new example mapping to the store."""
        with open(legal_text_path, 'r') as f:
            legal_text = f.read()
            
        shacl_graph = Graph()
        shacl_graph.parse(str(shacl_shape_path), format='turtle')
        
        annotations = None
        if annotations_path and annotations_path.exists():
            with open(annotations_path, 'r') as f:
                annotations = yaml.safe_load(f)
                
        self.examples.append(ExampleMapping(
            legal_text=legal_text,
            shacl_shape=shacl_graph,
            annotations=annotations
        ))
        
    def save_example(self, example: ExampleMapping, name: str) -> None:
        """Save an example mapping to disk."""
        example_dir = self.examples_dir / name
        example_dir.mkdir(parents=True, exist_ok=True)
        
        # Save legal text
        with open(example_dir / 'legal_text.txt', 'w') as f:
            f.write(example.legal_text)
            
        # Save SHACL shape
        example.shacl_shape.serialize(
            destination=str(example_dir / 'shape.ttl'),
            format='turtle'
        )
        
        # Save annotations if present
        if example.annotations:
            with open(example_dir / 'annotations.yaml', 'w') as f:
                yaml.dump(example.annotations, f)
                
    def load_all_examples(self) -> None:
        """Load all examples from the examples directory."""
        for example_dir in self.examples_dir.iterdir():
            if not example_dir.is_dir():
                continue
                
            legal_text_path = example_dir / 'legal_text.txt'
            shacl_path = example_dir / 'shape.ttl'
            annotations_path = example_dir / 'annotations.yaml'
            
            if legal_text_path.exists() and shacl_path.exists():
                self.add_example(
                    legal_text_path=legal_text_path,
                    shacl_shape_path=shacl_path,
                    annotations_path=annotations_path if annotations_path.exists() else None
                ) 

    def delete_example(self, index: int) -> None:
        """Delete an example at the given index and its associated files."""
        if 0 <= index < len(self.examples):
            example = self.examples[index]
            
            # Get the example directory
            example_dir = self.examples_dir / f"example_{index}"
            if example_dir.exists():
                # Delete all files in the directory
                for file in example_dir.iterdir():
                    file.unlink()
                # Remove the directory
                example_dir.rmdir()
            
            # Remove from list
            self.examples.pop(index) 