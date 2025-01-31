from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, XSD
import yaml
from pathlib import Path

from .datafields import DataFieldRegistry, DataField

@dataclass
class CitizenInstance:
    instance_id: str
    properties: Dict[str, Any]
    graph: Graph

class InstanceStore:
    def __init__(self, store_dir: Path, field_registry: DataFieldRegistry):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.field_registry = field_registry
        self.instances: Dict[str, CitizenInstance] = {}
        self.FF = Namespace("https://foerderfunke.org/default#")
        
    def create_instance(self, instance_id: str, properties: Dict[str, Any]) -> CitizenInstance:
        """Create a new Citizen instance with the given properties."""
        # Clean instance_id to be URI-safe
        safe_id = instance_id.replace(" ", "_").replace("-", "_").lower()
        
        # Validate property values against field constraints
        for field_name, value in properties.items():
            field = self.field_registry.get_field(field_name)
            if not field:
                raise ValueError(f"Unknown field: {field_name}")
            if not self._validate_value(field, value):
                raise ValueError(f"Invalid value for {field_name}: {value}")
        
        # Create RDF graph for the instance
        g = Graph()
        g.bind('ff', self.FF)
        
        instance_uri = self.FF[f"citizen_{safe_id}"]
        g.add((instance_uri, RDF.type, self.FF.Citizen))
        
        for field_name, value in properties.items():
            field = self.field_registry.get_field(field_name)
            # Convert the field path to use FF namespace
            if field.path.startswith('ff:'):
                predicate = self.FF[field.path[3:]]  # Remove 'ff:' prefix
            else:
                predicate = URIRef(field.path)
            literal_value = self._to_literal(value, field.datatype)
            g.add((instance_uri, predicate, literal_value))
        
        instance = CitizenInstance(safe_id, properties, g)
        self.instances[safe_id] = instance
        self._save_instance(instance)
        return instance
    
    def _validate_value(self, field: DataField, value: Any) -> bool:
        """Validate a value against a field's constraints."""
        if not value and field.constraints.get('minCount', 0) > 0:
            return False
            
        if field.datatype.startswith('xsd:'):
            datatype = field.datatype.split(':')[1]
            try:
                if datatype == 'integer':
                    int(value)
                elif datatype == 'decimal':
                    float(value)
                elif datatype == 'boolean':
                    return value in (True, False, 'true', 'false')
                elif datatype == 'date':
                    # Add date validation
                    pass
            except (ValueError, TypeError):
                return False
                
        if 'pattern' in field.constraints:
            import re
            if not re.match(field.constraints['pattern'], str(value)):
                return False
                
        return True
    
    def _to_literal(self, value: Any, datatype: str) -> Literal:
        """Convert a value to an RDF Literal with the correct datatype."""
        if datatype == 'xsd:integer':
            return Literal(int(value), datatype=XSD.integer)
        elif datatype == 'xsd:decimal':
            return Literal(float(value), datatype=XSD.decimal)
        elif datatype == 'xsd:boolean':
            return Literal(bool(value), datatype=XSD.boolean)
        elif datatype == 'xsd:date':
            return Literal(value, datatype=XSD.date)
        else:
            return Literal(str(value))
    
    def _save_instance(self, instance: CitizenInstance):
        """Save instance to disk."""
        instance_dir = self.store_dir / instance.instance_id
        instance_dir.mkdir(exist_ok=True)
        
        # Save properties
        with open(instance_dir / 'properties.yaml', 'w') as f:
            yaml.dump(instance.properties, f)
        
        # Save RDF graph
        instance.graph.serialize(destination=str(instance_dir / 'instance.ttl'), format='turtle')
    
    def load_all_instances(self):
        """Load all instances from disk."""
        for instance_dir in self.store_dir.iterdir():
            if not instance_dir.is_dir():
                continue
            
            instance_id = instance_dir.name
            
            # Load properties
            with open(instance_dir / 'properties.yaml', 'r') as f:
                properties = yaml.safe_load(f)
            
            # Load graph
            g = Graph()
            g.parse(instance_dir / 'instance.ttl', format='turtle')
            
            self.instances[instance_id] = CitizenInstance(instance_id, properties, g)
    
    def validate_instance(self, instance_id: str, shape: Graph) -> Tuple[bool, List[str]]:
        """Validate an instance against a SHACL shape."""
        from pyshacl import validate
        
        instance = self.instances.get(instance_id)
        if not instance:
            raise ValueError(f"Unknown instance: {instance_id}")
        
        conforms, results_graph, results_text = validate(
            instance.graph,
            shacl_graph=shape,
            inference='none'
        )
        
        # Extract validation messages
        messages = []
        if not conforms:
            for result in results_graph.subjects(RDF.type, URIRef("http://www.w3.org/ns/shacl#ValidationResult")):
                message = results_graph.value(result, URIRef("http://www.w3.org/ns/shacl#resultMessage"))
                if message:
                    messages.append(str(message))
        
        return conforms, messages
    
    def delete_instance(self, instance_id: str) -> None:
        """Delete an instance and its associated files."""
        if instance_id not in self.instances:
            raise ValueError(f"Unknown instance: {instance_id}")
            
        # Remove files
        instance_dir = self.store_dir / instance_id
        if instance_dir.exists():
            # Delete all files in the directory
            for file in instance_dir.iterdir():
                file.unlink()
            # Remove the directory
            instance_dir.rmdir()
        
        # Remove from memory
        del self.instances[instance_id] 