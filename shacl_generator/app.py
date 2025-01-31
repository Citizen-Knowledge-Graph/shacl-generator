import streamlit as st
from pathlib import Path
import rdflib
from rdflib import Graph
import tempfile
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io
import datetime

from shacl_generator.examples import ExampleStore, ExampleMapping
from shacl_generator.generator import ShaclGenerator, GeneratorContext
from shacl_generator.datafields import DataFieldRegistry, DataField
from shacl_generator.instances import InstanceStore
from shacl_generator.llm import SHACL_GENERATION_SYSTEM_PROMPT, RULE_EXTRACTION_SYSTEM_PROMPT
from shacl_generator.shapes import ShapeStore

import tiktoken

# Set page config first
st.set_page_config(
    page_title="Legal Text to SHACL Mapper",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Initialize paths
WORKSPACE_DIR = Path(__file__).parent.parent
EXAMPLES_DIR = WORKSPACE_DIR / "examples"
CONTEXT_PATH = WORKSPACE_DIR / "generator_context.json"
FIELDS_PATH = WORKSPACE_DIR / "datafields.yaml"
EXAMPLES_DIR.mkdir(exist_ok=True)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(io.BytesIO(pdf_file.read()))
    text = []
    for page in pdf_reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)

def truncate_text(text: str, max_length: int = 30000) -> str:
    """Truncate text to max_length while preserving paragraph structure."""
    if len(text) <= max_length:
        return text
    
    # Find the last paragraph break before max_length
    truncated = text[:max_length]
    last_break = truncated.rfind('\n\n')
    if (last_break == -1):
        last_break = truncated.rfind('\n')
    if (last_break == -1):
        last_break = truncated.rfind('. ')
    
    if (last_break != -1):
        truncated = truncated[:last_break]
    
    return truncated + "\n\n[Text truncated due to length...]"

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text using GPT tokenizer."""
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    return len(encoding.encode(text))

# Initialize components
@st.cache_resource
def init_components():
    example_store = ExampleStore(EXAMPLES_DIR)
    example_store.load_all_examples()
    
    field_registry = DataFieldRegistry(FIELDS_PATH)
    
    instance_store = InstanceStore(WORKSPACE_DIR / "instances", field_registry)
    instance_store.load_all_instances()
    
    shape_store = ShapeStore(WORKSPACE_DIR / "shapes")
    
    context = GeneratorContext.load(CONTEXT_PATH) if CONTEXT_PATH.exists() else GeneratorContext()
    generator = ShaclGenerator(context, example_store=example_store, field_registry=field_registry)
    return example_store, generator, field_registry, instance_store, shape_store

example_store, generator, field_registry, instance_store, shape_store = init_components()

st.title("Legal Text to SHACL Shape Mapper")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    mode = st.radio(
        "Mode",
        ["Generate SHACL", "Extract Rules","Manage Shapes", "Manage Examples", "Manage Guidelines", 
         "Manage Data Fields", "Manage Instances", "Manage Feedback"]
    )

if mode == "Generate SHACL":
    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.header("Legal Text Input")
        
        # Add file upload option
        uploaded_file = st.file_uploader("Upload PDF or paste text", type=['pdf', 'txt'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/pdf":
                    legal_text = extract_text_from_pdf(uploaded_file)
                else:  # txt file
                    legal_text = uploaded_file.read().decode()
                
                # Show extracted text with option to edit
                legal_text = st.text_area(
                    "Review and edit extracted text if needed",
                    value=legal_text,
                    height=300
                )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                legal_text = ""
        else:
            legal_text = st.text_area(
                "Or paste legal text directly",
                height=300
            )
        
        # Display token count
        if legal_text:
            token_count = count_tokens(legal_text)
            st.write(f"Token count: **{token_count:,}**")

        if st.button("Generate SHACL Shape"):
            if legal_text:
                with st.spinner("Generating SHACL shape..."):
                    # Generate unique ID for the text
                    text_id = str(hash(legal_text))
                    
                    # Create debug container
                    debug_container = st.empty()
                    
                    # Store old print function
                    old_print = print
                    debug_output = []
                    
                    # Create custom print function
                    def custom_print(*args):
                        message = " ".join(map(str, args))
                        debug_output.append(message)
                        debug_container.text("\n".join(debug_output))
                        old_print(*args)  # Still print to console
                    
                    # Replace print function
                    import builtins
                    builtins.print = custom_print
                    
                    try:
                        # Get the prompts
                        shacl_prompt = generator.llm._create_generation_prompt(
                            legal_text=legal_text,
                            examples=generator._get_relevant_examples(text_id),
                            feedback_history=generator._get_relevant_feedback(text_id),
                            guidelines=generator.context.general_guidelines
                        )
                        
                        # Store prompts in session state
                        st.session_state['current_prompt'] = shacl_prompt
                        
                        # Generate the shape and extract rules
                        shape, new_fields = generator.generate_shape(legal_text, text_id)
                        
                        st.session_state['current_shape'] = shape
                        st.session_state['current_text_id'] = text_id
                        
                        if new_fields:
                            st.info(f"Added {len(new_fields)} new data fields")
                            for field in new_fields:
                                st.write(f"- {field.name} ({field.datatype})")
                        
                        # Store debug output
                        st.session_state['debug_output'] = "\n".join(debug_output)
                        
                        # Store the generated shape
                        shape_store.add_shape(
                            shape_id=text_id,
                            legal_text=legal_text,
                            graph=shape,
                            description="Generated from legal text"
                        )
                        
                        st.success("SHACL shape generated!")
                    finally:
                        # Restore print function
                        builtins.print = old_print
            else:
                st.warning("Please provide some legal text first.")
    
    with col2:
        st.header("Generated SHACL Shape")
        
        if 'current_shape' in st.session_state:
            # Add tabs for shape, prompts, validation, and debug
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "SHACL Shape", 
                "System Prompt", 
                "Generation Prompt", 
                "Validate Instance",
                "Debug Output"
            ])
            
            with tab1:
                shape_text = st.session_state['current_shape'].serialize(format='turtle')
                st.text_area("SHACL Shape", shape_text, height=300)
                
                # Add feedback section under the shape
                feedback = st.text_area(
                    "Enter feedback to improve the shape", 
                    height=100,
                    key="shape_feedback"
                )
                
                if st.button("Improve Shape"):
                    with st.spinner("Improving SHACL shape..."):
                        improved_shape, new_fields = generator.improve_shape(
                            st.session_state['current_shape'],
                            feedback,
                            st.session_state['current_text_id']
                        )
                        # Save feedback and update shape
                        generator.context.add_feedback(
                            st.session_state['current_text_id'],
                            feedback,
                            improved_shape.serialize(format='turtle')
                        )
                        generator.context.save(CONTEXT_PATH)
                        st.session_state['current_shape'] = improved_shape
                        
                        if new_fields:
                            st.info(f"Added {len(new_fields)} new data fields")
                            for field in new_fields:
                                st.write(f"- {field.name} ({field.datatype})")
                        
                        st.success("Shape improved based on feedback!")
                        st.rerun()
            
            with tab2:
                st.text_area("System Prompt", 
                            SHACL_GENERATION_SYSTEM_PROMPT,
                            height=500)
            
            with tab3:
                st.text_area("Generation Prompt", 
                            st.session_state['current_prompt'],
                            height=500)
            
            with tab4:
                selected_instance = st.selectbox(
                    "Select Instance to Validate",
                    options=list(instance_store.instances.keys()),
                    key="validate_instance_select"
                )
                
                if selected_instance and st.button("Validate Instance"):
                    try:
                        conforms, messages = instance_store.validate_instance(
                            selected_instance,
                            st.session_state['current_shape']
                        )
                        
                        if conforms:
                            st.success(f"✅ Instance {selected_instance} conforms to the shape!")
                        else:
                            st.error("❌ Validation failed:")
                            for msg in messages:
                                st.write(f"- {msg}")
                                
                        # Show the instance details
                        with st.expander("View Instance Details"):
                            instance = instance_store.instances[selected_instance]
                            st.json(instance.properties)
                            st.text_area(
                                "Instance Graph",
                                instance.graph.serialize(format='turtle'),
                                height=200
                            )
                            
                    except Exception as e:
                        st.error(f"Error during validation: {str(e)}")
            
            with tab5:
                if 'debug_output' in st.session_state:
                    st.text(st.session_state['debug_output'])

elif mode == "Extract Rules":
    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.header("Legal Text Input")
        
        # Add file upload option
        uploaded_file = st.file_uploader("Upload PDF or paste text", type=['pdf', 'txt'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/pdf":
                    legal_text = extract_text_from_pdf(uploaded_file)
                else:  # txt file
                    legal_text = uploaded_file.read().decode()
                
                # Show extracted text with option to edit
                legal_text = st.text_area(
                    "Review and edit extracted text if needed",
                    value=legal_text,
                    height=300
                )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                legal_text = ""
        else:
            legal_text = st.text_area(
                "Or paste legal text directly",
                height=300
            )
        
        # Display token count
        if legal_text:
            token_count = count_tokens(legal_text)
            st.write(f"Token count: **{token_count:,}**")

        if st.button("Generate rules"):
            if legal_text:
                with st.spinner("Generating rules..."):
                    # Generate unique ID for the text
                    text_id = str(hash(legal_text))
                    
                    # Create debug container
                    debug_container = st.empty()
                    
                    # Store old print function
                    old_print = print
                    debug_output = []
                    
                    # Create custom print function
                    def custom_print(*args):
                        message = " ".join(map(str, args))
                        debug_output.append(message)
                        debug_container.text("\n".join(debug_output))
                        old_print(*args)  # Still print to console
                    
                    # Replace print function
                    import builtins
                    builtins.print = custom_print
                    
                    try:
                        # Get the prompts
                        rules_prompt = generator.llm._create_rules_generation_prompt(
                            legal_text=legal_text
                        )
                        
                        # Store prompts in session state
                        st.session_state['current_rules_prompt'] = rules_prompt
                        
                        # Generate the shape and extract rules
                        rules = generator.generate_rules(legal_text, text_id)
                        
                        st.session_state['current_rules'] = rules
                        st.session_state['current_text_id'] = text_id
                        
                        # Store debug output
                        st.session_state['debug_output'] = "\n".join(debug_output)                
                        
                        st.success("Rules generated!")
                    finally:
                        # Restore print function
                        builtins.print = old_print
            else:
                st.warning("Please provide some legal text first.")

    with col2:
        st.header("Extracted rules")
        
        if 'current_rules' in st.session_state:
            # Add tabs for shape, prompts, validation, and debug
            tab1, tab2, tab3, tab4 = st.tabs([
                "Rules", 
                "System Prompt", 
                "Generation Prompt", 
                "Debug Output"
            ])
            
            with tab1:
                shape_text = st.session_state['current_rules']
                st.text_area("Extracted Rules", shape_text, height=300)
            
            with tab2:
                st.text_area("System Prompt", 
                            RULE_EXTRACTION_SYSTEM_PROMPT,
                            height=500)
            
            with tab3:
                st.text_area("Generation Prompt", 
                            st.session_state['current_rules_prompt'],
                            height=500)
            
            with tab4:
                if 'debug_output' in st.session_state:
                    st.text(st.session_state['debug_output'])        

elif mode == "Manage Shapes":
    st.header("SHACL Shape Management")
    
    # View existing shapes
    st.subheader("Existing Shapes")
    for shape_id, shape in shape_store.shapes.items():
        with st.expander(f"Shape: {shape_id}"):
            col1, col2 = st.columns([5,1])
            
            with col1:
                st.text(f"Created: {shape.created_at}")
                st.text(f"Updated: {shape.updated_at}")
                if shape.description:
                    st.text(f"Description: {shape.description}")
            
            with col2:
                if st.button("🗑️ Delete", key=f"delete_shape_{shape_id}"):
                    shape_store.delete_shape(shape_id)
                    st.success(f"Shape {shape_id} deleted!")
                    st.rerun()
            
            # Show legal text and shape
            st.text_area("Legal Text", shape.legal_text, height=200, key=f"legal_text_{shape_id}")
            st.text_area("SHACL Shape", shape.graph.serialize(format='turtle'), height=300, key=f"shape_{shape_id}")
            
            # Add update functionality
            new_description = st.text_input("New Description", 
                                          value=shape.description or "", 
                                          key=f"description_{shape_id}")
            if st.button("Update Description", key=f"update_shape_{shape_id}"):
                shape_store.update_shape(shape_id, shape.graph, new_description)
                st.success("Shape updated!")
                st.rerun()

elif mode == "Manage Examples":
    st.header("Example Management")
    
    # Add new example
    with st.expander("Add New Example"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Add option for multiple legal texts
            num_texts = st.number_input("Number of input texts", min_value=1, value=1)
            legal_texts = []
            
            for i in range(num_texts):
                st.subheader(f"Input Text {i+1}")
                # Add file upload for example legal text
                uploaded_text_file = st.file_uploader(
                    f"Upload legal text {i+1} (PDF/TXT)",
                    type=['pdf', 'txt'],
                    key=f"example_text_{i}"
                )
                
                if uploaded_text_file is not None:
                    try:
                        if uploaded_text_file.type == "application/pdf":
                            text = extract_text_from_pdf(uploaded_text_file)
                        else:  # txt file
                            text = uploaded_text_file.read().decode()
                        
                        # Truncate if too long
                        text = truncate_text(text)
                    except Exception as e:
                        st.error(f"Error processing file {i+1}: {str(e)}")
                        text = ""
                else:
                    text = ""
                
                text = st.text_area(
                    f"Review and edit legal text {i+1}",
                    value=text,
                    height=200,
                    key=f"text_area_{i}"
                )
                legal_texts.append(text)
            
            with col2:
                # Add option to either upload or paste SHACL
                shape_input_method = st.radio(
                    "SHACL Shape Input Method",
                    ["Upload File", "Paste Text"]
                )
                
                if shape_input_method == "Upload File":
                    uploaded_shape = st.file_uploader(
                        "SHACL Shape (Turtle format)",
                        type=['ttl'],
                        key="example_shape"
                    )
                    shape_content = uploaded_shape.read().decode() if uploaded_shape else None
                else:
                    shape_content = st.text_area(
                        "Paste SHACL Shape (Turtle format)",
                        height=200,
                        key="pasted_shape"
                    )
                
                annotations = st.text_area(
                    "Annotations (Optional)", 
                    placeholder="Add any notes or explanations about this mapping",
                    height=100
                )
            
            if st.button("Save Example"):
                if any(legal_texts) and shape_content:
                    # Create temporary files
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmpdir = Path(tmpdir)
                        
                        # Save all legal texts
                        text_paths = []
                        for i, text in enumerate(legal_texts):
                            if text:  # Only save non-empty texts
                                text_path = tmpdir / f"text_{i}.txt"
                                with open(text_path, 'w') as f:
                                    f.write(text)
                                text_paths.append(text_path)
                        
                        # Save shape
                        shape_path = tmpdir / "shape.ttl"
                        with open(shape_path, 'w') as f:
                            f.write(shape_content)
                        
                        # Save annotations if provided
                        annotations_path = None
                        if annotations:
                            annotations_path = tmpdir / "annotations.yaml"
                            with open(annotations_path, 'w') as f:
                                f.write(annotations)
                        
                        # Add example for each text
                        for text_path in text_paths:
                            example_store.add_example(text_path, shape_path, annotations_path)
                        
                        st.success(f"Added {len(text_paths)} examples successfully!")
                else:
                    st.warning("Please provide at least one legal text and SHACL shape.")
    
    # View existing examples
    st.subheader("Existing Examples")
    for i, example in enumerate(example_store.examples):
        with st.expander(f"Example {i+1}"):
            # Add delete button in the header
            col_title, col_delete = st.columns([5,1])
            with col_title:
                st.subheader(f"Example {i+1}")
            with col_delete:
                if st.button("🗑️ Delete", key=f"delete_example_{i}"):
                    example_store.delete_example(i)  # We'll need to add this method
                    st.success(f"Example {i+1} deleted!")
                    st.rerun()  # Refresh the page to show updated list
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_area(
                    "Legal Text", 
                    example.legal_text, 
                    height=200,
                    key=f"view_text_{i}"
                )
            
            with col2:
                st.text_area(
                    "SHACL Shape", 
                    example.shacl_shape.serialize(format='turtle'),
                    height=200,
                    key=f"view_shape_{i}"
                )
            
            if example.annotations:
                st.text_area(
                    "Annotations", 
                    str(example.annotations), 
                    height=100,
                    key=f"view_annotations_{i}"
                )

elif mode == "Manage Guidelines":
    st.header("Manage Guidelines")
    
    # Add new guideline
    new_guideline = st.text_area(
        "Add New Guideline", 
        placeholder="Enter a new guideline for SHACL shape generation",
        height=100
    )
    if st.button("Add Guideline"):
        if new_guideline:
            generator.add_general_guideline(new_guideline)
            generator.context.save(CONTEXT_PATH)
            st.success("Guideline added successfully!")
    
    # View and manage existing guidelines
    st.subheader("Existing Guidelines")
    for i, guideline in enumerate(generator.context.general_guidelines):
        col1, col2 = st.columns([5,1])
        
        with col1:
            st.text_area(f"Guideline {i+1}", guideline, height=100, key=f"guideline_{i}")
            
        with col2:
            if st.button("🗑️ Delete", key=f"delete_guideline_{i}"):
                generator.context.remove_guideline(i)  # We'll add this method
                generator.context.save(CONTEXT_PATH)
                st.success(f"Guideline {i+1} deleted!")
                st.rerun()

elif mode == "Manage Instances":
    st.header("Instance Management")
    
    # Create new instance
    with st.expander("Create New Instance"):
        instance_id = st.text_input("Instance ID", placeholder="e.g., john_doe_1")
        
        properties = {}
        for field in field_registry.fields.values():
            # Check if field has allowed values in constraints
            if field.constraints and 'allowed_values' in field.constraints:
                # Create options list from allowed values
                options = [val['id'] for val in field.constraints['allowed_values']]
                labels = {val['id']: val.get('label', val['id']) 
                         for val in field.constraints['allowed_values']}
                
                value = st.selectbox(
                    f"{field.name}",
                    options=options,
                    format_func=lambda x: labels[x],
                    help=field.description
                )
            else:
                # Handle other datatypes as before
                if field.datatype == 'xsd:string':
                    value = st.text_input(f"{field.name}", help=field.description)
                elif field.datatype == 'xsd:integer':
                    value = st.number_input(f"{field.name}", help=field.description, step=1)
                elif field.datatype == 'xsd:decimal':
                    value = st.number_input(f"{field.name}", help=field.description)
                elif field.datatype == 'xsd:boolean':
                    value = st.checkbox(f"{field.name}", help=field.description)
                elif field.datatype == 'xsd:date':
                    value = st.date_input(
                        f"{field.name}", 
                        help=field.description,
                        min_value=datetime.date(1900, 1, 1),  # Allow dates from 1900
                        max_value=datetime.date.today()  # Up to today
                    )
            
            if value:
                properties[field.name] = value
        
        if st.button("Create Instance"):
            try:
                instance = instance_store.create_instance(instance_id, properties)
                st.success(f"Created instance {instance_id}")
            except ValueError as e:
                st.error(str(e))
    
    # View and validate instances
    st.subheader("Existing Instances")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_instance = st.selectbox(
            "Select Instance",
            options=list(instance_store.instances.keys())
        )
        
        if selected_instance:
            # Add delete button
            col_info, col_delete = st.columns([5,1])
            with col_delete:
                if st.button("🗑️ Delete", key=f"delete_instance_{selected_instance}"):
                    try:
                        instance_store.delete_instance(selected_instance)
                        st.success(f"Instance {selected_instance} deleted!")
                        st.rerun()  # Refresh the page
                    except ValueError as e:
                        st.error(str(e))
            
            with col_info:
                st.json(instance_store.instances[selected_instance].properties)
                
                st.text_area(
                    "RDF Graph",
                    instance_store.instances[selected_instance].graph.serialize(format='turtle'),
                    height=200
                )

elif mode == "Manage Feedback":
    st.header("Feedback Management")
    
    if not generator.context.feedback_history:
        st.info("No feedback history available yet.")
    else:
        for i, feedback_item in enumerate(generator.context.feedback_history):
            with st.expander(f"Feedback {i+1} (Text ID: {feedback_item.text_id})"):
                col1, col2 = st.columns([5,1])
                
                with col1:
                    st.text_area(
                        "Feedback",
                        feedback_item.feedback,
                        height=100,
                        key=f"feedback_{i}"
                    )
                    
                    st.text_area(
                        "Improved Shape",
                        feedback_item.improved_shape,
                        height=200,
                        key=f"improved_shape_{i}"
                    )
                
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_feedback_{i}"):
                        generator.context.remove_feedback(i)  # We'll add this method
                        generator.context.save(CONTEXT_PATH)
                        st.success(f"Feedback {i+1} deleted!")
                        st.rerun()

else:  # Manage Data Fields mode
    st.header("Manage Data Fields")
    
    # Import from SHACL
    with st.expander("Import Fields from SHACL"):
        st.write("""
        Upload or paste a SHACL file containing field definitions. 
        The system will extract property shapes and add them to the registry.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_shacl = st.file_uploader(
                "Upload SHACL file",
                type=['ttl'],
                key="field_definitions"
            )
            
            if uploaded_shacl:
                shacl_content = uploaded_shacl.read().decode()
                if st.button("Import Fields from File"):
                    try:
                        imported_fields = field_registry.import_from_shacl(shacl_content)
                        if imported_fields:
                            st.success(f"Successfully imported {len(imported_fields)} fields: {', '.join(imported_fields)}")
                        else:
                            st.warning("No valid property shapes found in the file.")
                    except Exception as e:
                        st.error(f"Error importing fields: {str(e)}")
        
        with col2:
            shacl_text = st.text_area(
                "Or paste SHACL content",
                height=200,
                placeholder="Paste your SHACL Turtle content here..."
            )
            
            if shacl_text and st.button("Import Fields from Text"):
                try:
                    imported_fields = field_registry.import_from_shacl(shacl_text)
                    if imported_fields:
                        st.success(f"Successfully imported {len(imported_fields)} fields: {', '.join(imported_fields)}")
                    else:
                        st.warning("No valid property shapes found in the text.")
                except Exception as e:
                    st.error(f"Error importing fields: {str(e)}")
    
    # View and edit existing fields
    st.subheader("Existing Data Fields")
    
    for field_name, field in field_registry.fields.items():
        with st.expander(f"Field: {field_name}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.text(f"Path: {field.path}")
                st.text_area("Description", field.description, key=f"desc_{field_name}")
            
            with col2:
                # Datatype selection
                datatype_options = [
                    "xsd:string",
                    "xsd:integer",
                    "xsd:decimal",
                    "xsd:boolean",
                    "xsd:date"
                ]
                new_datatype = st.selectbox(
                    "Datatype",
                    options=datatype_options,
                    index=datatype_options.index(field.datatype),
                    key=f"datatype_{field_name}"
                )
                
                if new_datatype != field.datatype:
                    if st.button("Update Datatype", key=f"update_{field_name}"):
                        try:
                            field_registry.update_field_datatype(field_name, new_datatype)
                            field_registry.save()
                            st.success(f"Updated datatype for {field_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error updating datatype: {str(e)}")
            
            # Show constraints
            if field.constraints:
                st.json(field.constraints)