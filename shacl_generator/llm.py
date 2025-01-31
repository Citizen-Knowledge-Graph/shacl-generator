from typing import Optional, List, Dict, Tuple
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from rdflib import Graph, Namespace, URIRef
import re

from .datafields import DataFieldRegistry, DataField

# Load environment variables
load_dotenv()

SHACL_GENERATION_SYSTEM_PROMPT = """
Your task is to complete the SHACL shapes graph below from texts describing the eligibility requirements for a social benefit. The shapes graph will be used to validate RDF user graphs containing personal information, ensuring that only individuals eligible for the given benefit conform to the shapes graph.

# GENERAL SHACL FORMAT RULES
1. Start with ALL necessary prefix declarations in this order:
   @prefix ff: <https://foerderfunke.org/default#> .
   @prefix sh: <http://www.w3.org/ns/shacl#> .
   @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
   @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
   @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

2. Use proper Turtle syntax:
   - End each prefix declaration with a dot (.).
   - Use semicolons (;) for multiple properties of the same subject.
   - Use dots (.) to separate statements.
   - Use semicolons (;) to separate multiple `sh:property` blocks.
   - Use commas (,) only for multiple RDF types within `a`, never between properties.
   - Maintain consistent indentation for readability.   

3. Output ONLY the Turtle syntax, no explanations or markdown.


# STRUCTURAL FORMAT OF PRIMARY SHAPE
1. Define a `RequirementProfile` with a succinct name in the `ff` namespace:
   Example: ff:buergergeld a ff:RequirementProfile .

2. Define a `MainPersonShape` with a name combining the benefit name and 'MainPersonShape':
   Example: ff:buergergeldMainPersonShape a sh:NodeShape, ff:EligibilityConstraint .

3. Link the `RequirementProfile` to the `MainPersonShape` using `ff:hasMainPersonShape`:
   Example: ff:buergergeld ff:hasMainPersonShape ff:buergergeldMainPersonShape .

4. Set the `MainPersonShape` target class:
   Example: ff:buergergeldMainPersonShape sh:targetClass ff:Citizen .

Example structure:
ff:buergergeld a ff:RequirementProfile ;
    ff:hasMainPersonShape ff:buergergeldMainPersonShape .

ff:buergergeldMainPersonShape a sh:NodeShape, ff:EligibilityConstraint ;
    sh:targetClass ff:Citizen .


# STRUCTURAL FORMAT OF PROPERTY CONSTRAINTS

1. Define each condition as a separate `sh:property` block with a **named URI instead of a blank node**.
2. Include cardinality constraints for each property:
   - Use `sh:minCount 1` for required properties.
   - Use `sh:maxCount` if there is an upper limit.
3. Use `sh:datatype` for data type constraints and `sh:pattern` for string patterns when applicable.
4. Add `sh:description` to explain each constraint in plain language.
5. Ensure each `sh:property` block is explicitly assigned a unique named URI (e.g., `ff:prop_aufenthaltsort, ff:prop_kindergeld`) instead of relying on blank nodes (`[]`). This prevents RDFLib from merging multiple properties into a list structure upon parsing.

Example of a correctly structured property shape:
ff:buergergeldMainPersonShape sh:property ff:prop_aufenthaltsort ;
    sh:property ff:prop_kindergeld .

ff:prop_aufenthaltsort sh:path ff:aufenthaltsort ;
    sh:in (ff:aufenthaltsort-ao-innerhalb) ;
    sh:description "The child must reside in Germany or an EU country, Iceland, Liechtenstein, Norway, or Switzerland." .

ff:prop_kindergeld sh:path ff:kindergeld ;
    sh:not [ sh:in (true) ] ;
    sh:description "Child benefit cannot be received if comparable benefits are received from an international or foreign institution." .


# LOGICAL HANDLING OF ELIGIBILITY CONDITIONS

1. Use distinct `sh:property` constraints for separate eligibility conditions. Do not group unrelated conditions.

Example of separate constraints:
ff:prop_income sh:path ff:income ;
       sh:maxInclusive 1500 ;
       sh:description "The person's income must not exceed 1500 EUR." .

ff:prop_children sh:path ff:hasChildren ;
       sh:minCount 1 ;
       sh:description "The person must have at least one dependent child." .

2. **Logical Relationships**
   - Use `sh:or`, `sh:and`, or `sh:not` **only when necessary**.
   - If a condition requires multiple factors, **separate them into distinct constraints**.

   Correct logical relationship:
   ff:prop_eligibilityCondition sh:path ff:eligibilityCondition ;
       sh:and (
           [ sh:path ff:kinder_18_21 ; sh:in (true) ]
           [ sh:path ff:kinder_arbeitslos ; sh:in (true) ]
       ) .
       
3. Handling Logical Dependencies with Conditional Fields**
  - Do not assume conditions exist in the same field if they require **separate validation**.
  - If an eligibility rule depends on multiple conditions, use **sh:or** inside an anonymous node (`[]`) to enforce dependencies.
  - Use `sh:and` inside `sh:or` when a condition requires **multiple sub-conditions to be true simultaneously**.
  - Create a **primary field** for the base condition (e.g., age range).
  - Create a **secondary conditional field** that **only applies when the primary condition is met** (e.g., employment or education status).t**.

   Correct Example: Expressing Logical Dependencies for Child Benefit Eligibility
   ff:prop_kinder_berechtigt sh:path ff:kinder_berechtigt ;
    sh:or (
        # Condition 1: Children under 18
        [ sh:path ff:kinder_18 ; sh:in (true) ]

        # Condition 2: Children aged 18-21 and unemployed
        [ sh:and (
            [ sh:path ff:kinder_18_21 ; sh:in (true) ]
            [ sh:path ff:kinder_arbeitslos ; sh:in (true) ]
        )]

        # Condition 3: Children aged 18-25 and in education
        [ sh:and (
            [ sh:path ff:kinder_18_25 ; sh:in (true) ]
            [ sh:path ff:kinder_ausbildung ; sh:in (true) ]
        )]
    ) .

   - This ensures that **child benefit eligibility** is only granted if at least one of these conditions is met:
     1. The child is **under 18** (`ff:kinder_18`).
     2. The child is **between 18 and 21 and unemployed** (`ff:kinder_18_21` and `ff:kinder_arbeitslos`).
     3. The child is **between 18 and 25 and in education** (`ff:kinder_18_25` and `ff:kinder_ausbildung`).


# FIELD USAGE EXAMPLE

1. Precision in Data Field Usage
   - Always use an existing data field if it exactly matches the needed condition.
   - Do not overuse existing data fields if they do not match the requirement.
   - If no suitable field exists, create a new one in the ff namespace.

   Correct use of existing data fields:
   ff:prop_aufenthaltsort sh:path ff:aufenthaltsort ;
       sh:in (ff:aufenthaltsort-ao-innerhalb) ;
       sh:description "The child must reside in Germany or an EU country, Iceland, Liechtenstein, Norway, or Switzerland." .

   Incorrect example (reusing an existing field incorrectly):
   ff:prop_kindergeld sh:path ff:kindergeld ;
       sh:not [ sh:in ( true ) ] ;
       sh:description "Child benefit cannot be received if comparable benefits are received from an international or foreign institution." .

   Correct fix (creating a new field instead):
   ff:prop_foreign_kindergeld sh:path ff:foreign_kindergeld ;
       sh:not [ sh:in ( true ) ] ;
       sh:description "Child benefit cannot be received if comparable benefits are received from an international or foreign institution." .

2. Explicit New Field Creation Guidelines
   - If a condition **does not fit an existing field**, create a new one.
   - Follow **namespace conventions**: `ff:<category>_<specific_condition>`.
   - Example:
     ff:prop_foreign_kindergeld sh:path ff:foreign_kindergeld ;
         sh:not [ sh:in ( true ) ] ;
         sh:description "Child benefit cannot be received if comparable benefits are received from an international or foreign institution." .

3. With existing datafield only use allowed values
  If a field `einkommen_neu` exists with:
  - path: ff:einkommen_neu
  - allowed_values: ["einkommen_neu-ao-selbstaendig", "einkommen_neu-ao-angestellt"]

Use it like this:
ff:prop_einkommen_neu sh:path ff:einkommen_neu ;
       sh:in (ff:einkommen_neu-ao-selbstaendig ff:einkommen_neu-ao-angestellt) ;
       sh:description "The person's income type must be either self-employed or employed." .

{additional_guidelines}

Analyze the legal text and create appropriate SHACL property shapes within the MainPersonShape.
Use a succinct, lowercase name for the benefit type in the ff namespace.
Output ONLY the Turtle syntax, no explanations."""

RULE_EXTRACTION_SYSTEM_PROMPT = """
You are a legal expert specialized in analyzing the requirements for people to receive social benefits. Your task is to:
1. Read the provided legal text carefully
2. Extract specific requirements
3. Present each requirements as a clear, concise statement
4. Focus on obligations, permissions, prohibitions, and conditions
5. Avoid vague language and ensure each rule is specific and measurable where possible
6. Use German as the primary language for the extracted rules

Format rules as a numbered list, with one rule per line. Each rule should be:
- Clear and unambiguous
- Self-contained (understandable without context)
- Actionable or measurable
- Focused on a single requirement"""

class LLMInterface:
    def __init__(self, model: str = "gpt-4o", field_registry: Optional[DataFieldRegistry] = None):
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
        """Create the generation prompt with all available context."""
        prompt_parts = [
            "Generate a SHACL shape for the following legal text.",
            "\nIMPORTANT: You MUST use these exact prefix declarations in this order:",
            "@prefix ff: <https://foerderfunke.org/default#> .",
            "@prefix sh: <http://www.w3.org/ns/shacl#> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "\nUse the 'ff:' prefix for all foerderfunke terms, NEVER use 'ns1:' or any other prefix."
        ]
        
        # Add available fields section with detailed information
        if self.field_registry:
            prompt_parts.append("\nAvailable data fields that MUST be used when applicable:")
            for field in self.field_registry.fields.values():
                field_info = [f"\nField: {field.name}"]
                field_info.append(f"Path: {field.path}")
                field_info.append(f"Datatype: {field.datatype}")
                field_info.append(f"Description: {field.description}")
                if field.constraints:
                    if 'allowed_values' in field.constraints:
                        values = [val['id'] for val in field.constraints['allowed_values']]
                        field_info.append(f"Allowed values: {', '.join(values)}")
                prompt_parts.append("\n".join(field_info))
        
        # Add examples if available
        if examples:
            prompt_parts.append("\nReference examples:")
            for i, example in enumerate(examples, 1):
                prompt_parts.extend([
                    f"\nExample {i}:",
                    f"Text: {example['text']}",
                    f"Shape: {example['shape']}"
                ])
        
        # Add feedback history if available
        if feedback_history:
            prompt_parts.append("\nPrevious feedback:")
            for item in feedback_history:
                prompt_parts.extend([
                    f"\nFeedback: {item['feedback']}",
                    f"Improved shape: {item['improved_shape']}"
                ])
        
        # Add guidelines
        if guidelines:
            prompt_parts.append("\nAdditional guidelines:")
            for guideline in guidelines:
                prompt_parts.append(f"- {guideline}")
        
        # Add the legal text last
        prompt_parts.extend([
            "\nLegal text to convert:",
            legal_text
        ])
        
        return "\n".join(prompt_parts)
    
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
    
    def _validate_and_fix_turtle(self, turtle_content: str) -> str:
        """Validate and fix common Turtle syntax issues."""
        # Fix common syntax issues
        fixed_content = turtle_content
        
        # Fix quotes in string literals (including unicode)
        fixed_content = re.sub(
            r'([a-zA-Z]+:(?:description|comment|label))\s+"([^"]*)"(?!\^\^)',
            r'\1 """\2"""',
            fixed_content
        )
        
        # Fix numeric literals with datatype
        fixed_content = re.sub(
            r'(\d+)\^\^xsd:(integer|decimal)',
            r'\1',  # Remove the datatype annotation when used with sh:maxInclusive etc.
            fixed_content
        )
        
        # Fix list syntax and trailing whitespace
        fixed_content = re.sub(
            r'\(\s*([^\)]+?)\s*\)',
            lambda m: '(' + ' '.join(m.group(1).split()) + ')',
            fixed_content
        )
        
        # Fix trailing whitespace in property blocks
        fixed_content = re.sub(r'\s+\]', ' ]', fixed_content)
        
        # Fix multiple consecutive newlines
        fixed_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', fixed_content)
        
        return fixed_content
    
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
        print("=== LLM OUTPUT ===")
        print(turtle_content)
        print("================")
        
        try:
            # Create graph and bind namespaces BEFORE parsing
            g = Graph()
            print("=== BEFORE PARSING ===")
            print("Namespaces:", list(g.namespaces()))
            
            g.parse(data=turtle_content, format='turtle')
            
            print("=== AFTER PARSING ===")
            print("Namespaces:", list(g.namespaces()))
            print("Graph content:")
            print(g.serialize(format='turtle'))
            print("==================")
            
            # Extract any new fields
            new_fields = []
            if self.field_registry:
                for name, datatype, description in self._extract_new_fields(turtle_content):
                    field = DataField(
                        name=name,
                        path=f"ex:{name}",
                        datatype=f"xsd:{datatype}",
                        description=description
                    )
                    new_fields.append(field)
            
            return g, new_fields
            
        except Exception as e:
            print("=== Trying to fix ===")
            # If parsing fails, try one more time with a fix request
            fix_prompt = f"""The following Turtle syntax is invalid. Please fix it to be valid Turtle/SHACL:

{turtle_content}

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
            # Create graph with bound namespaces
            g = Graph()
            g.bind('ff', "https://foerderfunke.org/default#")
            g.bind('sh', "http://www.w3.org/ns/shacl#")
            g.bind('xsd', "http://www.w3.org/2001/XMLSchema#")
            g.bind('rdfs', "http://www.w3.org/2000/01/rdf-schema#")
            g.bind('rdf', "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
            
            # Parse the improved shape into the graph
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

    def _create_rules_generation_prompt(self, legal_text: str) -> str:
        """Create the prompt for rule extraction."""
        prompt = f"""Please extract rules from the following legal text:

Legal Text:
{legal_text}

Extract clear, actionable rules and present them as a numbered list. Focus on:
- Requirements and obligations
- Permissions and rights
- Prohibitions and restrictions
- Conditions and qualifications
- Time limits and deadlines
- Specific measurements or quantities

Please list the rules:"""
        return prompt

    def generate_rules(self, legal_text: str) -> List[str]:
        """Generate a list of human-readable rules from legal text."""
        prompt = self._create_rules_generation_prompt(legal_text)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RULE_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        rules_text = response.choices[0].message.content
        print("=== LLM OUTPUT ===")
        print(rules_text)
        
        return rules_text