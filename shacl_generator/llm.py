from typing import Optional, List, Dict, Tuple
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from rdflib import Graph
import re

from .datafields import DataFieldRegistry, DataField

# Load environment variables
load_dotenv()

SHACL_GENERATION_SYSTEM_PROMPT = """You are a SHACL shape generator that converts legal text requirements into SHACL shapes.
Follow these structural requirements for all shapes:

1. Create a RequirementProfile with a succinct name in the ff namespace that reflects the benefit type
   Example: ff:buergergeld a ff:RequirementProfile ;

2. Create a MainPersonShape with a name that combines the benefit name with 'MainPersonShape'
   Example: ff:buergergeldMainPersonShape a sh:NodeShape, ff:EligibilityConstraint ;

3. Link them using ff:hasMainPersonShape
   Example: ff:buergergeld ff:hasMainPersonShape ff:buergergeldMainPersonShape .

4. Set the MainPersonShape target class
   Example: ff:buergergeldMainPersonShape sh:targetClass ff:Citizen .

Complete example structure:
@prefix ff: <http://example.org/foerderfunke#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .

ff:buergergeld a ff:RequirementProfile ;
    ff:hasMainPersonShape ff:buergergeldMainPersonShape .

ff:buergergeldMainPersonShape a sh:NodeShape, ff:EligibilityConstraint ;
    sh:targetClass ff:Citizen .

Additional guidelines:
- Use meaningful IDs for shapes and properties
- Add rdfs:label and rdfs:comment where appropriate
- Use existing vocabulary terms when possible
- Follow SHACL best practices for constraint definitions

{additional_guidelines}

Analyze the legal text and create appropriate SHACL property shapes within the MainPersonShape.
Use a succinct, lowercase name for the benefit type in the ff namespace.
"""

class LLMInterface:
    def __init__(self, model: str = "gpt-4o-mini", field_registry: Optional[DataFieldRegistry] = None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.field_registry = field_registry
        
    def _create_generation_prompt(
        self,
        legal_text: str,
        examples: List[Dict] = None,
        feedback_history: List[Dict] = None,
        guidelines: List[str] = None
    ) -> str:
        """Create a prompt for generating a SHACL shape from legal text."""
        prompt = [
            "You are a specialized AI that converts legal texts about social benefits into SHACL shapes.",
            "SHACL shapes formally describe the requirements that must be met for a person to be eligible for the benefit.",
            "\nYour task is to create a SHACL shape that captures all requirements from the following legal text.",
            "\nLEGAL TEXT:",
            legal_text,
            "\nGUIDELINES:",
            "1. Use sh:NodeShape to define the main shape for the person/applicant",
            "2. Use meaningful property paths that reflect the requirement's nature",
            "3. Include appropriate cardinality constraints (sh:minCount, sh:maxCount)",
            "4. Use sh:datatype for data type constraints",
            "5. Use sh:pattern for string patterns when applicable",
            "6. Add sh:description to explain each constraint in plain language",
            "\nIMPORTANT OUTPUT FORMAT RULES:",
            "1. Start with ALL necessary prefix declarations (@prefix)",
            "2. ALWAYS include these prefixes:",
            "   @prefix sh: <http://www.w3.org/ns/shacl#> .",
            "   @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "   @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "   @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "3. Use proper Turtle syntax with dots (.) after each statement",
            "4. End each prefix declaration with a dot (.)",
            "5. Use semicolons (;) for multiple properties of the same subject",
            "6. Output ONLY the Turtle syntax, no explanations or markdown"
        ]
        
        if self.field_registry:
            prompt.extend([
                "\nEXISTING DATA FIELDS:",
                "When defining properties, first check if there's a matching field below.",
                "Only create new property paths if no existing field matches the requirement.",
                self.field_registry.to_prompt_format()
            ])
        
        if guidelines:
            prompt.extend(["\nADDITIONAL GUIDELINES:"] + [f"- {g}" for g in guidelines])
            
        if examples:
            prompt.extend(["\nEXAMPLE MAPPINGS:"])
            for ex in examples:
                prompt.extend([
                    "\nInput text:",
                    ex["text"],
                    "\nSHACL shape:",
                    ex["shape"],
                    "\nAnnotations:",
                    ex["annotations"] if ex.get("annotations") else "None provided"
                ])
                
        if feedback_history:
            prompt.extend(["\nRELEVANT FEEDBACK FROM PREVIOUS GENERATIONS:"])
            for fb in feedback_history:
                prompt.extend([
                    f"\nFeedback: {fb['feedback']}",
                    f"Improved shape: {fb['improved_shape']}"
                ])
        
        return "\n".join(prompt)
    
    def _extract_new_fields(self, shape_text: str) -> List[Tuple[str, str, str]]:
        """Extract new field definitions from the shape text."""
        new_fields = []
        
        # Look for property paths and their descriptions
        path_pattern = r'ex:(\w+)\s+(?:a\s+sh:PropertyShape|sh:path\s+ex:\w+)\s*;[^;]*sh:datatype\s+xsd:(\w+)[^;]*(?:sh:description\s+"([^"]+)")?'
        matches = re.finditer(path_pattern, shape_text, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            field_name = match.group(1)
            datatype = match.group(2)
            description = match.group(3) or f"Field for {field_name}"
            
            # Only include fields that don't already exist
            if self.field_registry and not self.field_registry.get_field(field_name):
                new_fields.append((field_name, datatype, description))
        
        return new_fields
    
    def _create_improvement_prompt(
        self,
        current_shape: str,
        feedback: str,
        feedback_history: List[Dict] = None,
        guidelines: List[str] = None
    ) -> str:
        """Create a prompt for improving an existing SHACL shape based on feedback."""
        prompt = [
            "You are a specialized AI that improves SHACL shapes based on feedback.",
            "Your task is to modify the following SHACL shape according to the provided feedback.",
            "\nCURRENT SHAPE:",
            current_shape,
            "\nFEEDBACK TO ADDRESS:",
            feedback,
            "\nGUIDELINES:",
            "1. Preserve the existing structure where possible",
            "2. Only make changes that address the feedback",
            "3. Ensure the output remains valid Turtle syntax",
            "4. Add comments to explain significant changes"
        ]
        
        if self.field_registry:
            prompt.extend([
                "\nEXISTING DATA FIELDS:",
                "When modifying properties, first check if there's a matching field below.",
                "Only create new property paths if no existing field matches the requirement.",
                self.field_registry.to_prompt_format()
            ])
        
        if guidelines:
            prompt.extend(["\nADDITIONAL GUIDELINES:"] + [f"- {g}" for g in guidelines])
            
        if feedback_history:
            prompt.extend(["\nPREVIOUS FEEDBACK AND IMPROVEMENTS:"])
            for fb in feedback_history:
                prompt.extend([
                    f"\nFeedback: {fb['feedback']}",
                    f"Improved shape: {fb['improved_shape']}"
                ])
        
        return "\n".join(prompt)
    
    def _extract_turtle_content(self, text: str) -> str:
        """Extract valid Turtle content from the LLM response."""
        # Try to find content between triple backticks if present
        code_block_match = re.search(r"```(?:turtle)?\n(.*?)```", text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # If no code blocks, try to find content that starts with @prefix
        prefix_match = re.search(r"(@prefix.*$).*", text, re.MULTILINE | re.DOTALL)
        if prefix_match:
            return prefix_match.group(1).strip()
        
        # If no clear Turtle syntax indicators, return the whole text
        return text.strip()
    
    def _validate_and_fix_turtle(self, turtle_text: str) -> str:
        """Basic validation and fixing of Turtle syntax."""
        # Ensure required prefixes
        required_prefixes = {
            "sh": "<http://www.w3.org/ns/shacl#>",
            "rdf": "<http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "rdfs": "<http://www.w3.org/2000/01/rdf-schema#>",
            "xsd": "<http://www.w3.org/2001/XMLSchema#>",
            "ex": "<http://example.org/>"
        }
        
        lines = turtle_text.split("\n")
        existing_prefixes = set()
        
        # Find existing prefixes
        for line in lines:
            if line.startswith("@prefix"):
                prefix = line.split(":")[0].replace("@prefix", "").strip()
                existing_prefixes.add(prefix)
        
        # Add missing prefixes
        prefix_lines = []
        for prefix, uri in required_prefixes.items():
            if prefix not in existing_prefixes:
                prefix_lines.append(f"@prefix {prefix}: {uri} .")
        
        if prefix_lines:
            lines = prefix_lines + [""] + lines
        
        # Ensure statements end with dots
        fixed_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.endswith(".") and not line.endswith(";"):
                line = line + " ."
            fixed_lines.append(line)
        
        return "\n".join(fixed_lines)
    
    def generate_shape(
        self,
        legal_text: str,
        examples: List[Dict] = None,
        feedback_history: List[Dict] = None,
        guidelines: List[str] = None
    ) -> Tuple[Graph, List[DataField]]:
        """Generate a SHACL shape from legal text."""
        prompt = self._create_generation_prompt(
            legal_text=legal_text,
            examples=examples,
            feedback_history=feedback_history,
            guidelines=guidelines
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SHACL_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        # Extract and clean up the SHACL shape from the response
        shape_text = response.choices[0].message.content
        turtle_content = self._extract_turtle_content(shape_text)
        fixed_turtle = self._validate_and_fix_turtle(turtle_content)
        
        try:
            # Parse the shape into a Graph
            g = Graph()
            g.parse(data=fixed_turtle, format='turtle')
            
            # Extract any new fields
            new_fields = []
            if self.field_registry:
                for name, datatype, description in self._extract_new_fields(fixed_turtle):
                    field = DataField(
                        name=name,
                        path=f"ex:{name}",
                        datatype=f"xsd:{datatype}",
                        description=description
                    )
                    new_fields.append(field)
            
            return g, new_fields
            
        except Exception as e:
            # If parsing fails, try one more time with a fix request
            fix_prompt = f"""The following Turtle syntax is invalid. Please fix it to be valid Turtle/SHACL:

{fixed_turtle}

Output only the fixed Turtle syntax, no explanations."""
            
            fix_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Turtle/SHACL syntax expert. Fix the syntax issues in the provided Turtle content."},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.1
            )
            
            fixed_content = self._extract_turtle_content(fix_response.choices[0].message.content)
            g = Graph()
            g.parse(data=fixed_content, format='turtle')
            
            # Extract any new fields from the fixed content
            new_fields = []
            if self.field_registry:
                for name, datatype, description in self._extract_new_fields(fixed_content):
                    field = DataField(
                        name=name,
                        path=f"ex:{name}",
                        datatype=f"xsd:{datatype}",
                        description=description
                    )
                    new_fields.append(field)
            
            return g, new_fields
    
    def improve_shape(
        self,
        current_shape: Graph,
        feedback: str,
        feedback_history: List[Dict] = None,
        guidelines: List[str] = None
    ) -> Tuple[Graph, List[DataField]]:
        """Improve a SHACL shape based on feedback."""
        prompt = self._create_improvement_prompt(
            current_shape=current_shape.serialize(format='turtle'),
            feedback=feedback,
            feedback_history=feedback_history,
            guidelines=guidelines
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a specialized AI that improves SHACL shapes based on feedback. Output only valid Turtle syntax."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        # Extract and clean up the improved SHACL shape
        shape_text = response.choices[0].message.content
        turtle_content = self._extract_turtle_content(shape_text)
        fixed_turtle = self._validate_and_fix_turtle(turtle_content)
        
        try:
            # Parse the improved shape into a Graph
            g = Graph()
            g.parse(data=fixed_turtle, format='turtle')
            
            # Extract any new fields
            new_fields = []
            if self.field_registry:
                for name, datatype, description in self._extract_new_fields(fixed_turtle):
                    field = DataField(
                        name=name,
                        path=f"ex:{name}",
                        datatype=f"xsd:{datatype}",
                        description=description
                    )
                    new_fields.append(field)
            
            return g, new_fields
            
        except Exception as e:
            # If parsing fails, try one more time with a fix request
            fix_prompt = f"""The following Turtle syntax is invalid. Please fix it to be valid Turtle/SHACL:

{fixed_turtle}

Output only the fixed Turtle syntax, no explanations."""
            
            fix_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Turtle/SHACL syntax expert. Fix the syntax issues in the provided Turtle content."},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.1
            )
            
            fixed_content = self._extract_turtle_content(fix_response.choices[0].message.content)
            g = Graph()
            g.parse(data=fixed_content, format='turtle')
            
            # Extract any new fields from the fixed content
            new_fields = []
            if self.field_registry:
                for name, datatype, description in self._extract_new_fields(fixed_content):
                    field = DataField(
                        name=name,
                        path=f"ex:{name}",
                        datatype=f"xsd:{datatype}",
                        description=description
                    )
                    new_fields.append(field)
            
            return g, new_fields 