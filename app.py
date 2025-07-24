import os
import pandas as pd
import pickle
from pypdf import PdfReader
import re
import streamlit as st
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Resume Categorizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stats-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .category-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        word_vector = pickle.load(open("tfidf.pkl", "rb"))
        model = pickle.load(open("model.pkl", "rb"))
        return word_vector, model
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'tfidf.pkl' and 'model.pkl' are in the directory.")
        return None, None

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

def categorize_resumes(uploaded_files, output_directory, word_vector, model):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f'Processing: {uploaded_file.name}')
        
        if uploaded_file.name.endswith('.pdf'):
            try:
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                
                cleaned_resume = cleanResume(text)
                input_features = word_vector.transform([cleaned_resume])
                prediction_id = model.predict(input_features)[0]
                category_name = category_mapping.get(prediction_id, "Unknown")
                
                category_folder = os.path.join(output_directory, category_name)
                if not os.path.exists(category_folder):
                    os.makedirs(category_folder)
                
                target_path = os.path.join(category_folder, uploaded_file.name)
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                results.append({
                    'filename': uploaded_file.name, 
                    'category': category_name,
                    'file_size': f"{len(uploaded_file.getbuffer())/1024:.1f} KB",
                    'processed_at': datetime.now().strftime("%H:%M:%S")
                })
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text('Processing completed!')
    results_df = pd.DataFrame(results)
    return results_df

# Header
st.markdown('<h1 class="main-header">🤖 AI Resume Categorizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automatically categorize resumes using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Available Categories")
    
    # Display categories in an organized way
    categories = list(set(category_mapping.values()))
    categories.sort()
    
    col1, col2 = st.columns(2)
    mid_point = len(categories) // 2
    
    with col1:
        for cat in categories[:mid_point]:
            st.write(f"• {cat}")
    
    with col2:
        for cat in categories[mid_point:]:
            st.write(f"• {cat}")
    
    st.markdown("---")
    st.info("💡 **Tip**: Upload multiple PDF files for batch processing!")

# Load models
word_vector, model = load_models()

if word_vector is not None and model is not None:
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📁 Upload Resume Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Select one or more PDF resume files to categorize"
        )
        
        # Show file info if files are uploaded
        if uploaded_files:
            st.markdown("#### 📋 Selected Files:")
            for file in uploaded_files:
                file_size = len(file.getbuffer()) / 1024
                st.markdown(f"**{file.name}** - {file_size:.1f} KB")
    
    with col2:
        st.markdown("### ⚙️ Settings")
        output_directory = st.text_input(
            "Output Directory",
            "categorized_resumes",
            help="Directory where categorized resumes will be saved"
        )
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Statistics display
        if uploaded_files:
            st.markdown(f"""
            <div class="stats-container">
                <h3>📈 Upload Summary</h3>
                <p><strong>{len(uploaded_files)}</strong> files selected</p>
                <p><strong>{sum(len(f.getbuffer())/1024 for f in uploaded_files):.1f} KB</strong> total size</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Process button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Categorize Resumes",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files
        )
    
    # Processing logic
    if process_button:
        if uploaded_files and output_directory:
            with st.spinner('🔄 Processing resumes...'):
                results_df = categorize_resumes(uploaded_files, output_directory, word_vector, model)
            
            # Success message
            st.markdown("""
            <div class="success-message">
                <h4>✅ Processing Completed Successfully!</h4>
                <p>All resumes have been categorized and saved to their respective folders.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Results display
            st.markdown("### 📊 Categorization Results")
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", len(results_df))
            with col2:
                st.metric("Categories Found", results_df['category'].nunique())
            with col3:
                most_common = results_df['category'].mode()[0] if not results_df.empty else "N/A"
                st.metric("Most Common", most_common)
            with col4:
                success_rate = f"{(len(results_df) / len(uploaded_files)) * 100:.1f}%"
                st.metric("Success Rate", success_rate)
            
            # Detailed results table
            st.markdown("#### 📋 Detailed Results")
            st.dataframe(
                results_df,
                use_container_width=True,
                column_config={
                    "filename": "📄 File Name",
                    "category": "🏷️ Category",
                    "file_size": "📏 Size",
                    "processed_at": "⏰ Processed At"
                }
            )
            
            # Category breakdown
            st.markdown("#### 📈 Category Distribution")
            category_counts = results_df['category'].value_counts()
            st.bar_chart(category_counts)
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                results_csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=results_csv,
                    file_name=f'categorized_resumes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    type="secondary"
                )
            
            with col2:
                results_json = results_df.to_json(orient='records', indent=2).encode('utf-8')
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=results_json,
                    file_name=f'categorized_resumes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                    mime='application/json',
                    type="secondary"
                )
            
        else:
            st.error("❌ Please upload files and specify the output directory.")
    
    # Information section
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        ### About this Application
        
        This AI-powered resume categorizer uses **Machine Learning** to automatically classify resumes into different job categories:
        
        1. **Upload**: Select one or more PDF resume files
        2. **Process**: The AI analyzes the content using TF-IDF vectorization
        3. **Categorize**: Machine learning model predicts the most suitable job category
        4. **Organize**: Files are automatically sorted into category folders
        5. **Download**: Get detailed results in CSV or JSON format
        
        **Supported Categories**: {categories_count} different job categories including Software Development, Data Science, Engineering, HR, Sales, and more.
        
        **File Requirements**: PDF format only, no size limit per file.
        """.format(categories_count=len(category_mapping)))

else:
    st.error("⚠️ Cannot load ML models. Please check if 'tfidf.pkl' and 'model.pkl' files are available.")
    st.info("The application requires pre-trained models to function. Please ensure the model files are in the same directory as this script.")