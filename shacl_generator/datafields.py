from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from pathlib import Path
import yaml
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD, SH

@dataclass
class DataField:
    name: str  # The field name (e.g., "age", "income")
    path: str  # The property path in the SHACL shape
    datatype: str  # XSD datatype
    description: str  # Human-readable description
    examples: List[str] = field(default_factory=list)  # Example values or usage
    synonyms: List[str] = field(default_factory=list)  # Alternative terms that map to this field
    constraints: Dict[str, str] = field(default_factory=dict)  # Common constraints (e.g., min/max values)

class DataFieldRegistry:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.fields: Dict[str, DataField] = {}
        self.load()
        
    def import_from_shacl(self, shacl_content: str) -> List[str]:
        """
        Import data fields from a SHACL file.
        Returns a list of imported field names.
        """
        g = Graph()
        try:
            g.parse(data=shacl_content, format='turtle')
        except Exception as e:
            raise ValueError(f"Failed to parse SHACL content: {str(e)}")
        
        # Define namespaces
        sh = SH
        ff = Namespace("https://foerderfunke.org/default#")
        schema = Namespace("http://schema.org/")
        
        # First collect all answer options and their labels
        answer_options = {}
        for ao_uri in g.subjects(RDF.type, ff.AnswerOption):
            ao_id = str(ao_uri).split('#')[-1]
            # Try to get English label first, then any label
            label = None
            for l in g.objects(ao_uri, RDFS.label):
                if isinstance(l, Literal):
                    if l.language == 'en':
                        label = str(l)
                        break
                    elif not label:
                        label = str(l)
            answer_options[str(ao_uri)] = {
                'id': ao_id,
                'label': label or ao_id
            }
        
        print(f"Found {len(answer_options)} answer options")
        for uri, ao in answer_options.items():
            print(f"Answer option: {uri} -> {ao}")
        
        imported_fields: Set[str] = set()
        
        # Find all DataField instances
        for field_uri in g.subjects(RDF.type, ff.DataField):
            try:
                # Get field name from URI
                field_name = str(field_uri).split('#')[-1]
                
                # Get label and description, preferring English if available
                label = None
                for l in g.objects(field_uri, RDFS.label):
                    if isinstance(l, Literal):
                        if l.language == 'en':
                            label = l
                            break
                        elif not label:  # Take first label if no English
                            label = l
                
                description = None
                # Try rdfs:comment first
                for d in g.objects(field_uri, RDFS.comment):
                    if isinstance(d, Literal):
                        if d.language == 'en':
                            description = d
                            break
                        elif not description:
                            description = d
                
                # If no comment, try schema:question
                if not description:
                    for q in g.objects(field_uri, schema.question):
                        if isinstance(q, Literal):
                            if q.language == 'en':
                                description = q
                                break
                            elif not description:
                                description = q
                
                # Get category
                category = g.value(field_uri, schema.category, None, any=False)
                if category:
                    category = str(category).split('#')[-1]
                
                # Find constraints
                constraints = {}
                
                # Look for object constraints
                print(f"\nProcessing field: {field_name}")
                obj_constraints_list = list(g.objects(field_uri, ff.objectConstraints))
                print(f"Found {len(obj_constraints_list)} object constraints")
                
                for obj_constraints in obj_constraints_list:
                    print(f"Found object constraints: {obj_constraints}")
                    # For blank nodes, we need to look at their properties directly
                    if (obj_constraints, RDF.type, SH.PropertyShape) in g:
                        prop_shape = obj_constraints
                        print("Found property shape (direct)")
                    else:
                        # Try to find property shape through blank node
                        prop_shapes = list(g.subjects(RDF.type, SH.PropertyShape))
                        print(f"Found {len(prop_shapes)} property shapes through type")
                        prop_shape = None
                        for ps in prop_shapes:
                            if any(pred == RDF.type and obj == SH.PropertyShape 
                                 for pred, obj in g.predicate_objects(ps)):
                                prop_shape = ps
                                print(f"Found property shape through blank node: {ps}")
                                break
                    
                    if prop_shape:
                        print("Processing property shape")
                        # Get target objects
                        target_objects = g.value(prop_shape, SH.targetObjectsOf)
                        if target_objects:
                            print(f"Found targetObjectsOf: {target_objects}")
                            constraints['targetObjectsOf'] = str(target_objects)
                        
                        # Get datatype
                        datatype = g.value(prop_shape, SH.datatype)
                        if datatype:
                            print(f"Found datatype: {datatype}")
                            datatype_str = str(datatype)
                            if '#' in datatype_str:
                                datatype_str = 'xsd:' + datatype_str.split('#')[-1]
                            constraints['datatype'] = datatype_str
                        
                        # Get allowed values (sh:in)
                        in_prop = URIRef(SH + 'in')
                        print(f"Looking for sh:in with property: {in_prop}")
                        allowed_values = []
                        
                        def traverse_rdf_list(list_node):
                            """Helper function to traverse an RDF list and collect its items."""
                            items = []
                            try:
                                print(f"Traversing RDF list starting at: {list_node}")
                                # Get all triples involving this node
                                for p, o in g.predicate_objects(list_node):
                                    print(f"Found predicate-object: {p} -> {o}")
                                
                                while list_node:
                                    first = g.value(list_node, RDF.first)
                                    if first:
                                        print(f"Found list item: {first}")
                                        items.append(first)
                                    
                                    list_node = g.value(list_node, RDF.rest)
                                    print(f"Next node: {list_node}")
                                    
                                    if list_node == RDF.nil:
                                        break
                            except Exception as e:
                                print(f"Error traversing list: {e}")
                            return items
                        
                        # Get the sh:in value
                        in_values = list(g.objects(prop_shape, in_prop))
                        print(f"Direct sh:in values: {in_values}")
                        
                        for val in in_values:
                            if isinstance(val, BNode):
                                print(f"Found blank node for sh:in: {val}")
                                # Try to traverse as RDF list
                                items = traverse_rdf_list(val)
                                if items:
                                    allowed_values.extend(items)
                            else:
                                allowed_values.append(val)
                        
                        print(f"Final allowed values: {allowed_values}")
                        
                        if allowed_values:
                            value_options = []
                            for value in allowed_values:
                                value_uri = str(value)
                                print(f"Processing value: {value_uri}")
                                if value_uri in answer_options:
                                    print(f"Found in answer options: {answer_options[value_uri]}")
                                    value_options.append(answer_options[value_uri])
                                else:
                                    # Fallback if not found in answer options
                                    value_id = value_uri.split('#')[-1]
                                    print(f"Not found in answer options, using ID: {value_id}")
                                    value_options.append({
                                        'id': value_id,
                                        'label': value_id
                                    })
                            constraints['allowed_values'] = value_options
                
                # Look for usage constraints
                usage_constraints_list = list(g.objects(field_uri, ff.usageConstraints))
                print(f"Found {len(usage_constraints_list)} usage constraints")
                
                for usage_constraints in usage_constraints_list:
                    print(f"Found usage constraints: {usage_constraints}")
                    # For blank nodes, we need to look at their properties directly
                    if (usage_constraints, RDF.type, SH.NodeShape) in g:
                        node_shape = usage_constraints
                        print("Found node shape (direct)")
                    else:
                        # Try to find node shape through blank node
                        node_shapes = list(g.subjects(RDF.type, SH.NodeShape))
                        print(f"Found {len(node_shapes)} node shapes through type")
                        node_shape = None
                        for ns in node_shapes:
                            if any(pred == RDF.type and obj == SH.NodeShape 
                                 for pred, obj in g.predicate_objects(ns)):
                                node_shape = ns
                                print(f"Found node shape through blank node: {ns}")
                                break
                    
                    if node_shape:
                        print("Processing node shape")
                        # Get target subjects
                        target_subjects = g.value(node_shape, SH.targetSubjectsOf)
                        if target_subjects:
                            print(f"Found targetSubjectsOf: {target_subjects}")
                            constraints['targetSubjectsOf'] = str(target_subjects)
                        
                        # Look for property constraints
                        for prop in g.objects(node_shape, SH.property):
                            print(f"Found property constraint: {prop}")
                            # Get cardinality constraints
                            min_count = g.value(prop, SH.minCount)
                            if min_count:
                                print(f"Found minCount: {min_count}")
                                constraints['minCount'] = str(min_count)
                            max_count = g.value(prop, SH.maxCount)
                            if max_count:
                                print(f"Found maxCount: {max_count}")
                                constraints['maxCount'] = str(max_count)
                            
                            # Get path to verify it matches our field
                            path = g.value(prop, SH.path)
                            if path:
                                print(f"Found path: {path}")
                                if str(path) != str(field_uri):
                                    print(f"Path mismatch: {path} != {field_uri}")
                
                print(f"Final constraints for {field_name}: {constraints}")
                
                # Create field
                field = DataField(
                    name=field_name,
                    path=f"ff:{field_name}",
                    datatype=constraints.get('datatype', 'xsd:string'),
                    description=str(description) if description else (str(label) if label else f"Field for {field_name}"),
                    constraints=constraints
                )
                
                # If we have answer options with labels, add them as examples
                if 'allowed_values' in constraints:
                    field.examples = [f"{opt['label']} ({opt['id']})" for opt in constraints['allowed_values']]
                
                # Add field to registry
                self.add_field(field)
                imported_fields.add(field_name)
            
            except Exception as e:
                print(f"Warning: Failed to import field {field_uri}: {str(e)}")
                continue
        
        if not imported_fields:
            print("Warning: No valid data fields were found in the SHACL content")
        
        return list(imported_fields)
        
    def add_field(self, field: DataField) -> None:
        """Add a new data field to the registry."""
        self.fields[field.name] = field
        self.save()
        
    def get_field(self, name: str) -> Optional[DataField]:
        """Get a field by its exact name."""
        return self.fields.get(name)
    
    def find_matching_field(self, term: str, context: str = "") -> Optional[DataField]:
        """
        Find a matching field based on name, synonyms, and context.
        Args:
            term: The term to search for (e.g., "age", "years old")
            context: Additional context to help with matching
        """
        # First try exact matches
        for field in self.fields.values():
            if term.lower() == field.name.lower():
                return field
            if any(term.lower() == syn.lower() for syn in field.synonyms):
                return field
        
        # Then try partial matches
        for field in self.fields.values():
            if term.lower() in field.name.lower():
                return field
            if any(term.lower() in syn.lower() for syn in field.synonyms):
                return field
            if any(term.lower() in ex.lower() for ex in field.examples):
                return field
        
        return None
    
    def save(self) -> None:
        """Save the registry to disk."""
        data = {
            name: {
                "name": field.name,
                "path": field.path,
                "datatype": field.datatype,
                "description": field.description,
                "examples": field.examples,
                "synonyms": field.synonyms,
                "constraints": field.constraints
            }
            for name, field in self.fields.items()
        }
        
        with open(self.storage_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
    
    def load(self) -> None:
        """Load the registry from disk."""
        if not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r') as f:
            data = yaml.safe_load(f)
            
        if data:
            self.fields = {
                name: DataField(
                    name=field["name"],
                    path=field["path"],
                    datatype=field["datatype"],
                    description=field["description"],
                    examples=field["examples"],
                    synonyms=field["synonyms"],
                    constraints=field["constraints"]
                )
                for name, field in data.items()
            }
    
    def to_prompt_format(self) -> str:
        """Convert the registry to a format suitable for LLM prompts."""
        lines = ["Available data fields:"]
        
        for field in self.fields.values():
            field_desc = [
                f"\nField: {field.name}",
                f"Path: {field.path}",
                f"Type: {field.datatype}",
                f"Description: {field.description}"
            ]
            
            if field.examples:
                field_desc.append(f"Examples: {', '.join(field.examples)}")
            if field.synonyms:
                field_desc.append(f"Also known as: {', '.join(field.synonyms)}")
            if field.constraints:
                constraints = [f"{k}: {v}" for k, v in field.constraints.items()]
                field_desc.append(f"Constraints: {', '.join(constraints)}")
                
            lines.extend(field_desc)
            
        return "\n".join(lines)
    
    def suggest_new_field(self, term: str, context: str = "") -> DataField:
        """
        Suggest a new field structure based on a term and context.
        This is a helper method to create consistent new fields.
        """
        # Basic field name cleanup
        name = term.lower().replace(" ", "_").replace("-", "_")
        
        # Basic path creation (you might want to make this more sophisticated)
        path = f"ex:{name}"
        
        # Guess datatype based on common patterns
        datatype = "xsd:string"  # default
        if any(age_term in term.lower() for age_term in ["age", "years"]):
            datatype = "xsd:integer"
        elif any(amount_term in term.lower() for amount_term in ["amount", "income", "payment", "euro"]):
            datatype = "xsd:decimal"
        elif any(date_term in term.lower() for date_term in ["date", "time", "when"]):
            datatype = "xsd:date"
        elif any(bool_term in term.lower() for bool_term in ["is", "has", "can"]):
            datatype = "xsd:boolean"
            
        return DataField(
            name=name,
            path=path,
            datatype=datatype,
            description=f"Field for {term}",
            examples=[],
            synonyms=[term] if term != name else [],
            constraints={}
        )
    
    def update_field_datatype(self, field_name: str, new_datatype: str) -> None:
        """Update the datatype of an existing field."""
        if field_name not in self.fields:
            raise ValueError(f"Unknown field: {field_name}")
            
        if not new_datatype.startswith("xsd:"):
            raise ValueError("Datatype must start with 'xsd:'")
            
        # Update the field's datatype
        self.fields[field_name].datatype = new_datatype

    def to_string(self) -> str:
        str = []
        for field in self.fields.values():
            field_info = [f"\nField: {field.name}", f"Path: {field.path}", f"Datatype: {field.datatype}",
                          f"Description: {field.description}"]
            if field.constraints:
                if 'allowed_values' in field.constraints:
                    values = [val['id'] for val in field.constraints['allowed_values']]
                    field_info.append(f"Allowed values: {', '.join(values)}")
            str.append("\n".join(field_info))
        return str
