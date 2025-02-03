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

3. **Handling Nested Conditions**
   - For conditions where one property depends on another (e.g., age and asset limits), use **`sh:qualifiedValueShape`** or **`sh:node`** to explicitly define the dependency.
   - Always group dependent constraints together using `sh:and` or `sh:or` inside a `sh:node` or `sh:qualifiedValueShape`.

   Correct Example: Nested Conditions for Asset Limits Based on Age
   ff:prop_assets sh:path ff:assetsValue ;
       sh:datatype xsd:decimal ;
       sh:or (
           [ sh:description "The student's assets must not exceed 15,000 EUR if under 30 years old." ;
             sh:maxInclusive 15000 ;
             sh:property [
                 sh:path ff:hasAge ;
                 sh:maxExclusive 30 ;
             ]
           ]
           [ sh:description "The student's assets must not exceed 45,000 EUR if 30 years or older." ;
             sh:maxInclusive 45000 ;
             sh:property [
                 sh:path ff:hasAge ;
                 sh:minInclusive 30 ;
             ]
           ]
       ) .
       
   sh:or (
    [
      sh:description "If the student is under 30, their assets must not exceed 15,000 EUR." ;
      sh:property [
        sh:path ff:hasAge ;
        sh:maxExclusive 30 ;
      ] ;
      sh:property [
        sh:path ff:assetValue ;
        sh:datatype xsd:decimal ;
        sh:maxInclusive 15000 ;
      ]
    ]
    [
      sh:description "If the student is 30 or older, their assets must not exceed 45,000 EUR." ;
      sh:property [
        sh:path ff:hasAge ;
        sh:minInclusive 30 ;
      ] ;
      sh:property [
        sh:path ff:assetValue ;
        sh:datatype xsd:decimal ;
        sh:maxInclusive 45000 ;
      ]
    ]
  ) .    

4. **Avoid Overlapping Conditions**
   - Ensure that conditions in `sh:or` or `sh:and` are mutually exclusive or clearly defined to avoid ambiguity.
   - Use `sh:description` to clarify the intent of each condition.

5. **Use `sh:qualifiedValueShape` for Dependent Constraints**
   - If a condition depends on another property, use `sh:qualifiedValueShape` to define the dependency.

   Correct Example:
   ff:prop_independent_bafoeg sh:property [
       sh:path ff:apprenticeshipCompleted ;
       sh:qualifiedValueShape [
           sh:path ff:yearsWorked ;
           sh:minInclusive 3 ;
       ] ;
       sh:qualifiedMinCount 1 ;
       sh:message "If an apprenticeship was completed, the student must have worked at least 3 years."
   ] .

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
    
    def _process_llm_response(self, response_text: str) -> Tuple[Graph, List[DataField]]:
        """Process LLM response and return a Graph and any new fields."""
        # Extract turtle content from response
        turtle_content = self._extract_turtle_content(response_text)
        print("=== LLM OUTPUT ===")
        print(turtle_content)
        print("================")
        
        try:
            # Create graph and bind namespaces BEFORE parsing
            g = Graph()
            g.bind('ff', "https://foerderfunke.org/default#")
            g.bind('sh', "http://www.w3.org/ns/shacl#")
            g.bind('xsd', "http://www.w3.org/2001/XMLSchema#")
            g.bind('rdfs', "http://www.w3.org/2000/01/rdf-schema#")
            g.bind('rdf', "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
            
            print("=== BEFORE PARSING ===")
            print("Namespaces:", list(g.namespaces()))
            
            # Parse the content
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
            g.bind('ff', "https://foerderfunke.org/default#")
            g.bind('sh', "http://www.w3.org/ns/shacl#")
            g.bind('xsd', "http://www.w3.org/2001/XMLSchema#")
            g.bind('rdfs', "http://www.w3.org/2000/01/rdf-schema#")
            g.bind('rdf', "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
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
        
        return self._process_llm_response(response.choices[0].message.content)

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
        
        return self._process_llm_response(response.choices[0].message.content)

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
    
    
    def citique_agent(self, current_shape: Graph) -> Tuple[Graph, List[DataField]]:
        """Critique a SHACL shape and suggest improvements."""
        prompt = f"""
        Task:  
        Review the following SHACL shape and ensure that all **conditional dependencies** between properties are properly expressed using `sh:or`, `sh:and`, and `sh:not`.  

        Requirements:  
        1. **Explicit Conditional Logic**:  
        - If a condition depends on another property’s value (e.g., a requirement applies only under certain circumstances), model the dependency explicitly using `sh:and`, `sh:or`, or `sh:not`.  
        - Pay special attention to cases where multiple conditions must be met simultaneously (`sh:and`) or where alternative conditions exist (`sh:or`).  

        2. **Nested Conditions**:  
        - Ensure that nested conditions are properly structured. For example:  
            - Use `sh:or` to express **alternative conditions** (e.g., "either X or Y must be true").  
            - Use `sh:and` to express **combined conditions** (e.g., "both X and Y must be true").  
            - Use `sh:not` to express **disqualifying conditions** (e.g., "X must not be true").  

        3. **Consistency**:  
        - Ensure that all property constraints follow a consistent structure (e.g., using `sh:minCount`, `sh:maxCount`, or `sh:hasValue` uniformly).  
        - Do not change the overall structure of the SHACL shape (e.g., properties should remain listed under `sh:property`). Only update the constraints where necessary.  

        4. **Examples of Nested Conditions**:  
        - **Example 1**: A property requiring **either of two values** should use `sh:or`:  
            ```turtle
            sh:or (
                [ sh:hasValue true ; sh:path ff:conditionA ]
                [ sh:hasValue true ; sh:path ff:conditionB ]
            )
            ```  
        - **Example 2**: A property requiring **both a condition and a dependency** should use `sh:and`:  
            ```turtle
            sh:and (
                [ sh:hasValue true ; sh:path ff:conditionA ]
                [ sh:hasValue false ; sh:path ff:disqualifyingCondition ]
            )
            ```  
        - **Example 3**: A property that **only applies under certain conditions** should use `sh:not` or nested `sh:or`:  
            ```turtle
            sh:or (
                [ sh:not [ sh:hasValue true ; sh:path ff:disqualifyingCondition ] ]
                [ sh:hasValue true ; sh:path ff:specialCondition ]
            )
            ```  
        - **Example 4**: A property that **depends on another property** should use `sh:and` and `sh:or` to express the dependency:
           ```turtle
            sh:or (
                [
                    sh:description "Students under 25 must have a GPA of at least 3.5." ;
                    sh:and (
                        [ sh:path ff:age ; sh:maxExclusive 25 ]
                        [ sh:path ff:gpa ; sh:minInclusive 3.5 ]
                    )
                ]
                [
                    sh:description "Students 25 or older must have a GPA of at least 3.8." ;
                    sh:and (
                        [ sh:path ff:age ; sh:minInclusive 25 ]
                        [ sh:path ff:gpa ; sh:minInclusive 3.8 ]
                    )
                ]
            )

        5. **Output Format**:  
        - Return the updated SHACL shape in **Turtle syntax**.  
        - Include comments (`#`) to explain any changes made to the original shape.  

        CURRENT SHAPE:  
        {current_shape.serialize(format='turtle')}
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI specialized in critiquing SHACL shapes and suggesting improvements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._process_llm_response(response.choices[0].message.content)