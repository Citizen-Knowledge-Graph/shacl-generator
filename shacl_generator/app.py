import streamlit as st
from pathlib import Path
import rdflib
from rdflib import Graph
import tempfile
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io

from shacl_generator.examples import ExampleStore, ExampleMapping
from shacl_generator.generator import ShaclGenerator, GeneratorContext
from shacl_generator.datafields import DataFieldRegistry, DataField

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

# Initialize components
@st.cache_resource
def init_components():
    example_store = ExampleStore(EXAMPLES_DIR)
    example_store.load_all_examples()
    
    field_registry = DataFieldRegistry(FIELDS_PATH)
    
    context = GeneratorContext.load(CONTEXT_PATH) if CONTEXT_PATH.exists() else GeneratorContext()
    generator = ShaclGenerator(context, example_store=example_store, field_registry=field_registry)
    return example_store, generator, field_registry

example_store, generator, field_registry = init_components()

st.title("Legal Text to SHACL Shape Mapper")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    mode = st.radio(
        "Mode",
        ["Generate SHACL", "Manage Examples", "Manage Guidelines", "Manage Data Fields"]
    )

if mode == "Generate SHACL":
    col1, col2 = st.columns(2)
    
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
        
        if st.button("Generate SHACL Shape"):
            if legal_text:
                with st.spinner("Generating SHACL shape..."):
                    # Generate unique ID for the text
                    text_id = str(hash(legal_text))
                    shape, new_fields = generator.generate_shape(legal_text, text_id)
                    st.session_state['current_shape'] = shape
                    st.session_state['current_text_id'] = text_id
                    
                    if new_fields:
                        st.info(f"Added {len(new_fields)} new data fields")
                        for field in new_fields:
                            st.write(f"- {field.name} ({field.datatype})")
                    
                    st.success("SHACL shape generated!")
            else:
                st.warning("Please provide some legal text first.")
    
    with col2:
        st.header("Generated SHACL Shape")
        if 'current_shape' in st.session_state:
            shape_text = st.session_state['current_shape'].serialize(format='turtle')
            st.text_area("SHACL Shape", shape_text, height=300)
            
            # Feedback section
            st.subheader("Provide Feedback")
            feedback = st.text_area("Enter feedback to improve the shape", height=100)
            
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

elif mode == "Manage Examples":
    st.header("Example Management")
    
    # Add new example
    with st.expander("Add New Example"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Add file upload for example legal text
            uploaded_text_file = st.file_uploader(
                "Upload legal text (PDF/TXT)",
                type=['pdf', 'txt'],
                key="example_text"
            )
            
            if uploaded_text_file is not None:
                try:
                    if uploaded_text_file.type == "application/pdf":
                        legal_text = extract_text_from_pdf(uploaded_text_file)
                    else:  # txt file
                        legal_text = uploaded_text_file.read().decode()
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    legal_text = ""
            else:
                legal_text = ""
            
            legal_text = st.text_area(
                "Review and edit legal text",
                value=legal_text,
                height=200
            )
        
        with col2:
            uploaded_shape = st.file_uploader(
                "SHACL Shape (Turtle format)",
                type=['ttl'],
                key="example_shape"
            )
            annotations = st.text_area(
                "Annotations (Optional)", 
                placeholder="Add any notes or explanations about this mapping",
                height=100
            )
        
        if st.button("Save Example"):
            if legal_text and uploaded_shape:
                # Create temporary files
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)
                    
                    # Save legal text
                    text_path = tmpdir / "text.txt"
                    with open(text_path, 'w') as f:
                        f.write(legal_text)
                    
                    # Save shape
                    shape_path = tmpdir / "shape.ttl"
                    with open(shape_path, 'wb') as f:
                        f.write(uploaded_shape.read())
                    
                    # Save annotations if provided
                    annotations_path = None
                    if annotations:
                        annotations_path = tmpdir / "annotations.yaml"
                        with open(annotations_path, 'w') as f:
                            f.write(annotations)
                    
                    # Add example
                    example_store.add_example(text_path, shape_path, annotations_path)
                    st.success("Example added successfully!")
            else:
                st.warning("Please provide both legal text and SHACL shape.")
    
    # View existing examples
    st.subheader("Existing Examples")
    for i, example in enumerate(example_store.examples):
        with st.expander(f"Example {i+1}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_area("Legal Text", example.legal_text, height=200)
            
            with col2:
                st.text_area("SHACL Shape", 
                            example.shacl_shape.serialize(format='turtle'),
                            height=200)
            
            if example.annotations:
                st.text_area("Annotations", str(example.annotations), height=100)

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
    
    # View existing guidelines
    st.subheader("Existing Guidelines")
    for i, guideline in enumerate(generator.context.general_guidelines):
        st.text_area(f"Guideline {i+1}", guideline, height=100)

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
    
    # Add new field manually
    with st.expander("Add New Data Field"):
        col1, col2 = st.columns(2)
        
        with col1:
            field_name = st.text_input("Field Name", placeholder="e.g., age, income")
            field_path = st.text_input("Property Path", placeholder="e.g., ex:age")
            field_datatype = st.selectbox(
                "Data Type",
                ["xsd:string", "xsd:integer", "xsd:decimal", "xsd:date", "xsd:boolean"]
            )
        
        with col2:
            field_description = st.text_area(
                "Description",
                placeholder="Describe what this field represents"
            )
            field_examples = st.text_input(
                "Examples",
                placeholder="Comma-separated examples"
            )
            field_synonyms = st.text_input(
                "Synonyms",
                placeholder="Comma-separated alternative terms"
            )
        
        if st.button("Add Field"):
            if field_name and field_path and field_description:
                field = DataField(
                    name=field_name,
                    path=field_path,
                    datatype=field_datatype,
                    description=field_description,
                    examples=field_examples.split(",") if field_examples else [],
                    synonyms=field_synonyms.split(",") if field_synonyms else []
                )
                field_registry.add_field(field)
                st.success(f"Added field: {field_name}")
            else:
                st.warning("Please provide name, path, and description.")
    
    # View existing fields
    st.subheader("Existing Data Fields")
    for field in field_registry.fields.values():
        with st.expander(f"{field.name} ({field.datatype})"):
            st.write("**Path:**", field.path)
            st.write("**Description:**", field.description)
            
            # Display constraints in a structured way
            st.write("**Constraints:**")
            if not field.constraints:
                st.write("*No constraints defined*")
            else:
                # Object constraints
                if any(k in field.constraints for k in ['datatype', 'allowed_values', 'pattern', 'minInclusive', 'maxInclusive', 'minExclusive', 'maxExclusive', 'languageIn', 'uniqueLang']):
                    st.write("*Object Constraints:*")
                    if 'datatype' in field.constraints:
                        st.write(f"- Datatype: `{field.constraints['datatype']}`")
                    if 'allowed_values' in field.constraints:
                        st.write("- Allowed Values:")
                        for val in field.constraints['allowed_values']:
                            st.write(f"  - {val['label']} (`{val['id']}`)")
                    if 'pattern' in field.constraints:
                        st.write(f"- Pattern: `{field.constraints['pattern']}`")
                    for constraint in ['minInclusive', 'maxInclusive', 'minExclusive', 'maxExclusive']:
                        if constraint in field.constraints:
                            st.write(f"- {constraint}: {field.constraints[constraint]}")
                    if 'languageIn' in field.constraints:
                        st.write(f"- Language In: {', '.join(field.constraints['languageIn'])}")
                    if 'uniqueLang' in field.constraints:
                        st.write(f"- Unique Language: {field.constraints['uniqueLang']}")
                
                # Usage constraints
                if any(k in field.constraints for k in ['minCount', 'maxCount', 'qualifiedMinCount', 'qualifiedMaxCount', 'qualifiedValueShape']):
                    st.write("*Usage Constraints:*")
                    if 'minCount' in field.constraints:
                        st.write(f"- Minimum Count: {field.constraints['minCount']}")
                    if 'maxCount' in field.constraints:
                        st.write(f"- Maximum Count: {field.constraints['maxCount']}")
                    if 'qualifiedMinCount' in field.constraints:
                        st.write(f"- Qualified Minimum Count: {field.constraints['qualifiedMinCount']}")
                    if 'qualifiedMaxCount' in field.constraints:
                        st.write(f"- Qualified Maximum Count: {field.constraints['qualifiedMaxCount']}")
                    if 'qualifiedValueShape' in field.constraints:
                        st.write(f"- Qualified Value Shape: {field.constraints['qualifiedValueShape']}")
                
                # Target constraints
                if any(k in field.constraints for k in ['targetObjectsOf', 'targetSubjectsOf']):
                    st.write("*Target Constraints:*")
                    if 'targetObjectsOf' in field.constraints:
                        st.write(f"- Target Objects Of: {field.constraints['targetObjectsOf']}")
                    if 'targetSubjectsOf' in field.constraints:
                        st.write(f"- Target Subjects Of: {field.constraints['targetSubjectsOf']}")
            
            if field.examples:
                st.write("**Examples:**", ", ".join(field.examples))
            if field.synonyms:
                st.write("**Synonyms:**", ", ".join(field.synonyms)) 