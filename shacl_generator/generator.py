from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from rdflib import Graph, Namespace
from rdflib.namespace import XSD, RDFS

from .llm import LLMInterface
from .datafields import DataFieldRegistry, DataField

@dataclass
class FeedbackHistory:
    text_id: str
    feedback: str
    improved_shape: str
    
@dataclass
class GeneratorContext:
    """Maintains the context of previous interactions and feedback."""
    feedback_history: List[FeedbackHistory] = field(default_factory=list)
    general_guidelines: List[str] = field(default_factory=list)
    
    def add_feedback(self, text_id: str, feedback: str, improved_shape: str) -> None:
        self.feedback_history.append(FeedbackHistory(
            text_id=text_id,
            feedback=feedback,
            improved_shape=improved_shape
        ))
    
    def add_guideline(self, guideline: str) -> None:
        """Add a general guideline that should apply to all future generations."""
        self.general_guidelines.append(guideline)
    
    def remove_guideline(self, index: int) -> None:
        """Remove a guideline by its index."""
        if 0 <= index < len(self.general_guidelines):
            del self.general_guidelines[index]
        
    def remove_feedback(self, index: int) -> None:
        """Remove a feedback item by its index."""
        if 0 <= index < len(self.feedback_history):
            del self.feedback_history[index]
        
    def save(self, path: Path) -> None:
        """Save context to disk."""
        data = {
            'feedback_history': [
                {
                    'text_id': f.text_id,
                    'feedback': f.feedback,
                    'improved_shape': f.improved_shape
                }
                for f in self.feedback_history
            ],
            'general_guidelines': self.general_guidelines
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, path: Path) -> 'GeneratorContext':
        """Load context from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        context = cls()
        for feedback in data['feedback_history']:
            context.add_feedback(
                text_id=feedback['text_id'],
                feedback=feedback['feedback'],
                improved_shape=feedback['improved_shape']
            )
        context.general_guidelines.extend(data['general_guidelines'])
        return context

RULE_EXTRACTION_PROMPT = """Given the following legal text, extract the main rules and requirements in clear, human-readable sentences. Each rule should be concise and focus on a single requirement or constraint.

Legal Text:
{legal_text}

Examples of good rule extraction:
1. The minimum age requirement is 18 years.
2. Applicants must provide valid identification.
3. Applications must be submitted at least 30 days before the deadline.

Please extract the rules in a similar format:"""

class ShaclGenerator:
    def __init__(self, context: Optional[GeneratorContext] = None, example_store=None, field_registry: Optional[DataFieldRegistry] = None):
        self.context = context or GeneratorContext()
        self.example_store = example_store
        self.field_registry = field_registry
        self.llm = LLMInterface(field_registry=field_registry)
        
        # Define namespaces
        self.FF = Namespace("https://foerderfunke.org/default#")
        self.SH = Namespace("http://www.w3.org/ns/shacl#")
        self.RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        
    def _get_relevant_examples(self, text_id: str, max_examples: int = 3) -> List[Dict]:
        """Get relevant examples for the generation context."""
        if not self.example_store:
            return []
            
        examples = []
        for example in self.example_store.examples:
            examples.append({
                "text": example.legal_text,
                "shape": example.shacl_shape.serialize(format='turtle'),
                "annotations": example.annotations
            })
        return examples[:max_examples]  # TODO: Implement similarity-based selection
        
    def _get_relevant_feedback(self, text_id: str, max_items: int = 5) -> List[Dict]:
        """Get relevant feedback history for the generation context."""
        feedback_items = []
        for feedback in self.context.feedback_history:
            if feedback.text_id != text_id:  # Exclude feedback for current text
                feedback_items.append({
                    "feedback": feedback.feedback,
                    "improved_shape": feedback.improved_shape
                })
        return feedback_items[:max_items]  # TODO: Implement relevance-based selection
        
    def generate_shape(self, legal_text: str, text_id: str) -> Tuple[Graph, List[DataField]]:
        """Generate a SHACL shape from legal text."""
        # Initialize the graph with standard prefixes
        g = Graph()
        
        # Bind namespaces in the order we want them to appear
        g.bind('ff', self.FF)  # Bind ff first to make it the preferred namespace
        g.bind('sh', self.SH)
        g.bind('xsd', XSD)
        g.bind('rdfs', RDFS)
        g.bind('rdf', self.RDF)

        # Get examples and feedback for context
        examples = self._get_relevant_examples(text_id)
        feedback_history = self._get_relevant_feedback(text_id)
        
        # Generate the shape using LLM
        generated_graph, new_fields = self.llm.generate_shape(
            legal_text=legal_text,
            examples=examples,
            feedback_history=feedback_history,
            guidelines=self.context.general_guidelines
        )
        
        # Merge the generated graph into our base graph
        g += generated_graph
        
        # Add any new fields to the registry
        if self.field_registry and new_fields:
            for field in new_fields:
                self.field_registry.add_field(field)
        
        return g, new_fields
    
    def deploy_second_agent(self, shape: Graph) -> Tuple[Graph, List[DataField]]:
        """Deploy a second agent to generate a SHACL shape from legal text."""
        
        # Improve the shape using LLM
        improved_graph, new_fields = self.llm.critique_agent(current_shape=shape)
        
        # Add any new fields to the registry
        if self.field_registry and new_fields:
            for field in new_fields:
                self.field_registry.add_field(field)
        
        return improved_graph, new_fields
    
        
    def improve_shape(self, shape: Graph, feedback: str, text_id: str) -> Tuple[Graph, List[DataField]]:
        """Improve a SHACL shape based on feedback."""
        feedback_history = self._get_relevant_feedback(text_id)
        
        # Improve the shape using LLM
        improved_graph, new_fields = self.llm.improve_shape(
            current_shape=shape,
            feedback=feedback,
            feedback_history=feedback_history,
            guidelines=self.context.general_guidelines
        )
        
        # Add any new fields to the registry
        if self.field_registry and new_fields:
            for field in new_fields:
                self.field_registry.add_field(field)
        
        return improved_graph, new_fields
        
    def add_general_guideline(self, guideline: str) -> None:
        """Add a guideline that should apply to all future generations."""
        self.context.add_guideline(guideline)

    def extract_rules(self, legal_text: str, text_id: str) -> list[str]:
        """Extract human-readable rules from legal text."""
        prompt = RULE_EXTRACTION_PROMPT.format(legal_text=legal_text)
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a legal expert who extracts clear, concise rules from legal texts. Focus on actionable requirements and constraints."
        )
        
        # Process the response into a list of rules
        rules = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line[0] == '-'):
                # Remove leading numbers or bullets and clean up
                rule = line.split('.', 1)[-1].strip()
                rule = rule.lstrip('- ').strip()
                if rule:
                    rules.append(rule)
        
        return rules

    def generate_rules(self, legal_text: str) -> List[str]:
        """Generate human-readable rules from legal text."""
        # Generate rules using LLM
        return self.llm.generate_rules(legal_text)