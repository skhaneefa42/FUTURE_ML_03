"""
app.py - Complete Resume Screening System with Streamlit UI
All features: Resume ranking, skill extraction, skill gap analysis, visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from PyPDF2 import PdfReader
import io
import os
from datetime import datetime

# Download required NLTK data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    return " ".join(words)

# ==================== TEXT PREPROCESSOR ====================
class TextPreprocessor:
    """Handles text cleaning and preprocessing"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add custom stopwords
        custom_stopwords = {'experience', 'skills', 'work', 'job', 'role', 
                           'candidate', 'year', 'years', 'using', 'including'}
        self.stop_words.update(custom_stopwords)
    
    def clean_text(self, text):
        """Basic text cleaning"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, text):
        """Remove common stopwords"""
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

# ==================== SKILL EXTRACTOR ====================
class SkillExtractor:
    """Extracts technical skills from text with categorization"""
    
    def __init__(self):
        # Comprehensive skill database
        self.skills_db = {
            'Programming Languages': {
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust',
                'swift', 'kotlin', 'php', 'typescript', 'scala', 'r', 'matlab',
                'perl', 'html', 'css', 'bash', 'shell'
            },
            'ML & AI Frameworks': {
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'mxnet', 'caffe',
                'theano', 'jax', 'hugging face', 'transformers', 'langchain',
                'openai', 'llama', 'bert', 'gpt', 'machine learning', 'deep learning'
            },
            'ML Concepts': {
                'natural language processing', 'computer vision', 'reinforcement learning',
                'nlp', 'llm', 'large language models', 'generative ai', 'neural networks',
                'regression', 'classification', 'clustering', 'random forest', 'xgboost'
            },
            'Data Science & Analytics': {
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'scipy',
                'data visualization', 'statistics', 'data analysis', 'sql', 'tableau',
                'power bi', 'excel', 'looker', 'data mining', 'etl'
            },
            'Cloud Platforms': {
                'aws', 'amazon web services', 'azure', 'gcp', 'google cloud',
                'cloud computing', 'ec2', 's3', 'lambda', 'sagemaker', 'azure ml',
                'vertex ai', 'cloud run', 'kubernetes', 'docker'
            },
            'DevOps & MLOps': {
                'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'terraform',
                'ansible', 'prometheus', 'grafana', 'mlflow', 'kubeflow',
                'airflow', 'github actions', 'gitlab ci'
            },
            'Databases': {
                'sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'cassandra',
                'elasticsearch', 'dynamodb', 'bigquery', 'redshift', 'snowflake',
                'oracle', 'sqlite', 'couchdb'
            },
            'Soft Skills': {
                'leadership', 'communication', 'teamwork', 'problem solving',
                'critical thinking', 'project management', 'agile', 'scrum',
                'time management', 'collaboration', 'presentation'
            }
        }
        
        # Flatten skills for easier matching
        self.all_skills = set()
        for category in self.skills_db:
            self.all_skills.update(self.skills_db[category])
    
    def extract_skills(self, text):
        """Extract skills from text with categories"""
        text_lower = text.lower()
        extracted = {category: set() for category in self.skills_db.keys()}
        
        for category, skills in self.skills_db.items():
            for skill in skills:
                # Use word boundaries for exact matching
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    extracted[category].add(skill)
        
        return extracted
    
    def get_skill_list(self, text):
        """Return flat list of extracted skills"""
        extracted = self.extract_skills(text)
        all_skills = []
        for category in extracted:
            all_skills.extend(extracted[category])
        return list(set(all_skills))
    
    def compare_skills(self, resume_skills, job_skills):
        """Compare resume skills with job requirements"""
        resume_set = set(resume_skills)
        job_set = set(job_skills)
        
        matched = list(resume_set.intersection(job_set))
        missing = list(job_set - resume_set)
        
        match_percentage = (len(matched) / len(job_set)) * 100 if job_set else 0
        
        return matched, missing, match_percentage

# ==================== RESUME RANKER ====================
class ResumeRanker:
    """Ranks resumes based on job description similarity"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.job_vector = None
    
    def vectorize_texts(self, texts):
        """Convert texts to TF-IDF vectors"""
        return self.tfidf_vectorizer.fit_transform(texts)
    
    def calculate_similarity(self, job_text, resume_texts):
        """Calculate cosine similarity between job and resumes"""
        all_texts = [job_text] + resume_texts
        vectors = self.vectorize_texts(all_texts)
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        
        similarities = cosine_similarity(resume_vectors, job_vector.reshape(1, -1))
        return similarities.flatten()
    
    def rank_resumes(self, job_text, resume_texts, resume_names, resume_skills_lists, job_skills):
        """Complete ranking with skill analysis"""
        # Calculate similarity scores
        scores = self.calculate_similarity(job_text, resume_texts)
        
        # Create results dataframe
        results = pd.DataFrame({
            'Candidate': resume_names,
            'Similarity Score': scores,
            'Match Percentage': [round(s * 100, 2) for s in scores]
        })
        
        # Add skill information
        matched_skills_list = []
        missing_skills_list = []
        skill_match_percentages = []
        
        for resume_skills in resume_skills_lists:
            matched, missing, skill_pct = self.compare_skills_with_job(resume_skills, job_skills)
            matched_skills_list.append(matched)
            missing_skills_list.append(missing)
            skill_match_percentages.append(skill_pct)
        
        results['Matched Skills'] = matched_skills_list
        results['Missing Skills'] = missing_skills_list
        results['Skill Match %'] = skill_match_percentages
        
        # Calculate weighted score (60% text similarity, 40% skill match)
        results['Weighted Score'] = (
            0.6 * results['Similarity Score'] + 
            0.4 * (results['Skill Match %'] / 100)
        )
        
        # Rank candidates
        results['Rank'] = results['Weighted Score'].rank(ascending=False, method='min').astype(int)
        results = results.sort_values('Rank')
        
        return results
    
    def compare_skills_with_job(self, resume_skills, job_skills):
        """Helper method to compare skills"""
        resume_set = set(resume_skills)
        job_set = set(job_skills)
        
        matched = list(resume_set.intersection(job_set))
        missing = list(job_set - resume_set)
        percentage = (len(matched) / len(job_set)) * 100 if job_set else 0
        
        return matched, missing, percentage

# ==================== PDF PROCESSING ====================
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# ==================== MAIN APP ====================
def main():
    # Page configuration
    st.set_page_config(
        page_title="Resume Screening System",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .candidate-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }
        .skill-badge {
            display: inline-block;
            background-color: #e9ecef;
            padding: 0.2rem 0.5rem;
            border-radius: 5px;
            margin: 0.2rem;
            font-size: 0.8rem;
        }
        .missing-skill {
            background-color: #f8d7da;
            color: #721c24;
        }
        .matched-skill {
            background-color: #d4edda;
            color: #155724;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header"><h1>📄 AI-Powered Resume Screening System</h1><p>Intelligent candidate ranking with skill gap analysis</p></div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/resume.png", width=80)
        st.header("⚙️ Configuration")
        
        # Weightage settings
        st.subheader("Ranking Weights")
        text_weight = st.slider("Text Similarity Weight", 0.0, 1.0, 0.6, 0.1)
        skill_weight = st.slider("Skill Match Weight", 0.0, 1.0, 0.4, 0.1)
        
        st.markdown("---")
        st.subheader("📊 About")
        st.info(
            "This system uses:\n"
            "- TF-IDF Vectorization for text similarity\n"
            "- Cosine Similarity for ranking\n"
            "- Comprehensive skill database with 100+ skills\n"
            "- Weighted scoring for accurate matching"
        )
    
    # Main content area - two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Job Description")
        job_input_method = st.radio("Input method:", ["Paste text", "Upload file"])
        
        job_description = ""
        if job_input_method == "Paste text":
            job_description = st.text_area(
                "Paste job description here:",
                height=300,
                placeholder="Example:\n\nWe are looking for a Machine Learning Engineer with experience in Python, TensorFlow, and AWS..."
            )
        else:
            uploaded_job = st.file_uploader("Upload job description (TXT)", type=['txt'])
            if uploaded_job:
                job_description = uploaded_job.read().decode('utf-8')
                st.text_area("Job Description Preview:", job_description[:500] + "...", height=200)
        
        # Extract skills from job description
        if job_description:
            preprocessor = TextPreprocessor()
            skill_extractor = SkillExtractor()
            cleaned_job = preprocessor.preprocess(job_description)
            job_skills_dict = skill_extractor.extract_skills(cleaned_job)
            job_skills_list = skill_extractor.get_skill_list(cleaned_job)
            
            st.subheader("📋 Required Skills Extracted")
            for category, skills in job_skills_dict.items():
                if skills:
                    with st.expander(f"{category} ({len(skills)})"):
                        st.write(", ".join(sorted(skills)))
    
    with col2:
        st.header("👥 Resumes")
        upload_method = st.radio("Upload method:", ["Upload files", "Use sample data"])
        
        resume_texts = []
        resume_names = []
        resume_files = []
        
        if upload_method == "Upload files":
            uploaded_files = st.file_uploader(
                "Upload resumes (PDF or TXT)",
                type=['pdf', 'txt'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                for file in uploaded_files:
                    if file.name.endswith('.pdf'):
                        text = extract_text_from_pdf(file)
                    else:
                        text = file.read().decode('utf-8')
                    
                    resume_texts.append(text)
                    resume_names.append(file.name.replace('.pdf', '').replace('.txt', ''))
                    resume_files.append(file.name)
                
                st.success(f"✅ Uploaded {len(resume_texts)} resumes")
        
        else:  # Use sample data
            st.info("Using sample resumes for demonstration")
            # Sample resumes
            sample_resumes = {
                "John_DoE_ML_Engineer": """
                Name: John Doe
                Experience: 5 years
                Skills: Python, Machine Learning, TensorFlow, NLP, SQL, AWS, Docker
                Projects: Built ML models for customer churn prediction
                """,
                "Jane_Smith_Web_Dev": """
                Name: Jane Smith
                Experience: 3 years
                Skills: JavaScript, React, HTML, CSS, Node.js
                Projects: Developed multiple web applications
                """,
                "Mike_Johnson_Data_Scientist": """
                Name: Mike Johnson
                Experience: 7 years
                Skills: Python, Machine Learning, Deep Learning, PyTorch, SQL, AWS, Kubernetes
                Projects: End-to-end ML pipeline deployment
                """,
                "Sarah_Williams_Analyst": """
                Name: Sarah Williams
                Experience: 2 years
                Skills: SQL, Excel, Tableau, Basic Python
                Projects: Data analysis and visualization dashboards
                """
            }
            
            for name, content in sample_resumes.items():
                resume_texts.append(content)
                resume_names.append(name)
            
            st.success(f"✅ Loaded {len(resume_texts)} sample resumes")
    
    # Process and rank button
    if st.button("🚀 Screen and Rank Candidates", type="primary", use_container_width=True):
        if not job_description:
            st.error("Please provide a job description")
        elif not resume_texts:
            st.error("Please upload or load resumes")
        else:
            with st.spinner("Processing resumes..."):
                # Initialize components
                preprocessor = TextPreprocessor()
                skill_extractor = SkillExtractor()
                ranker = ResumeRanker()
                
                # Preprocess texts
                cleaned_job = preprocessor.preprocess(job_description)
                cleaned_resumes = [preprocessor.preprocess(resume) for resume in resume_texts]
                
                # Extract skills
                job_skills_list = skill_extractor.get_skill_list(cleaned_job)
                resume_skills_lists = [skill_extractor.get_skill_list(resume) for resume in cleaned_resumes]
                
                # Rank resumes with custom weights
                results = ranker.rank_resumes(
                    cleaned_job, cleaned_resumes, resume_names, 
                    resume_skills_lists, job_skills_list
                )
                
                # Store results in session state
                st.session_state['results'] = results
                st.session_state['job_skills'] = job_skills_list
                st.session_state['resume_skills'] = resume_skills_lists
                st.session_state['resume_texts'] = resume_texts
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        job_skills_list = st.session_state['job_skills']
        resume_skills_lists = st.session_state['resume_skills']
        resume_texts = st.session_state['resume_texts']
        
        st.markdown("---")
        st.header("📊 Screening Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candidates", len(results))
        with col2:
            st.metric("Top Score", f"{results.iloc[0]['Match Percentage']:.1f}%")
        with col3:
            avg_score = results['Match Percentage'].mean()
            st.metric("Average Score", f"{avg_score:.1f}%")
        with col4:
            best_skill_match = results['Skill Match %'].max()
            st.metric("Best Skill Match", f"{best_skill_match:.1f}%")
        
        # Visualization
        st.subheader("📈 Ranking Visualization")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart of match percentages
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results))]
        axes[0].barh(results['Candidate'], results['Match Percentage'], color=colors)
        axes[0].set_xlabel('Match Percentage (%)')
        axes[0].set_title('Overall Match Score by Candidate')
        axes[0].invert_yaxis()
        
        # Comparison chart
        x = np.arange(len(results))
        width = 0.35
        axes[1].bar(x - width/2, results['Similarity Score'] * 100, width, label='Text Similarity', color='#3498db')
        axes[1].bar(x + width/2, results['Skill Match %'], width, label='Skill Match', color='#e74c3c')
        axes[1].set_xlabel('Candidates')
        axes[1].set_ylabel('Score (%)')
        axes[1].set_title('Text Similarity vs Skill Match')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(results['Candidate'], rotation=45, ha='right')
        axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detailed results table
        st.subheader("🏆 Ranked Candidates")
        
        display_cols = ['Rank', 'Candidate', 'Match Percentage', 'Skill Match %', 'Weighted Score', 'Matched Skills', 'Missing Skills']
        st.dataframe(
            results[display_cols].style.background_gradient(subset=['Match Percentage', 'Skill Match %'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Detailed view for each candidate
        st.subheader("📋 Candidate Details")
        
        for idx, row in results.iterrows():
            with st.expander(f"#{row['Rank']} - {row['Candidate']} (Match: {row['Match Percentage']:.1f}%)"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📊 Scores:**")
                    st.markdown(f"- Overall Match: **{row['Match Percentage']:.1f}%**")
                    st.markdown(f"- Skill Match: **{row['Skill Match %']:.1f}%**")
                    st.markdown(f"- Weighted Score: **{row['Weighted Score']:.3f}**")
                    
                    st.markdown(f"**✅ Matched Skills ({len(row['Matched Skills'])}):**")
                    if row['Matched Skills']:
                        for skill in row['Matched Skills']:
                            st.markdown(f'<span class="skill-badge matched-skill">✓ {skill}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown("*No matching skills found*")
                    
                    st.markdown(f"**❌ Missing Skills ({len(row['Missing Skills'])}):**")
                    if row['Missing Skills']:
                        for skill in row['Missing Skills']:
                            st.markdown(f'<span class="skill-badge missing-skill">✗ {skill}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown("*No missing skills - Perfect match!*")
                
                with col2:
                    # Skill radar chart for this candidate
                    candidate_idx = results[results['Candidate'] == row['Candidate']].index[0]
                    candidate_skills = resume_skills_lists[candidate_idx]
                    
                    # Create skill comparison
                    skill_categories = ['Programming', 'ML/AI', 'Cloud', 'Databases', 'DevOps']
                    job_category_counts = []
                    candidate_category_counts = []
                    
                    skill_extractor_local = SkillExtractor()
                    job_skills_dict = skill_extractor_local.extract_skills(' '.join(job_skills_list))
                    candidate_skills_dict = skill_extractor_local.extract_skills(' '.join(candidate_skills))
                    
                    category_mapping = {
                        'Programming': 'Programming Languages',
                        'ML/AI': 'ML & AI Frameworks',
                        'Cloud': 'Cloud Platforms',
                        'Databases': 'Databases',
                        'DevOps': 'DevOps & MLOps'
                    }
                    
                    for cat in skill_categories:
                        actual_cat = category_mapping[cat]
                        job_category_counts.append(len(job_skills_dict.get(actual_cat, set())))
                        candidate_category_counts.append(len(candidate_skills_dict.get(actual_cat, set())))
                    
                    # Create radar chart
                    fig2, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection='polar'))
                    angles = np.linspace(0, 2 * np.pi, len(skill_categories), endpoint=False).tolist()
                    
                    # Close the plot
                    candidate_category_counts += candidate_category_counts[:1]
                    job_category_counts += job_category_counts[:1]
                    angles += angles[:1]
                    
                    ax.plot(angles, candidate_category_counts, 'o-', linewidth=2, label='Candidate', color='#2ecc71')
                    ax.plot(angles, job_category_counts, 'o-', linewidth=2, label='Job Required', color='#e74c3c')
                    ax.fill(angles, candidate_category_counts, alpha=0.25, color='#2ecc71')
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(skill_categories)
                    ax.set_ylim(0, max(max(job_category_counts), max(candidate_category_counts)) + 1)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    ax.set_title('Skill Profile Comparison', size=10)
                    
                    st.pyplot(fig2)
                
                # Resume preview
                with st.expander("View Resume Text"):
                    st.text(resume_texts[candidate_idx][:1000])
        
        # Download results
        st.subheader("💾 Download Results")
        csv = results[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"resume_screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Recommendations
        st.subheader("💡 Recommendations")
        best_candidate = results.iloc[0]
        if best_candidate['Missing Skills']:
            st.info(f"**Top candidate ({best_candidate['Candidate']})** matches {len(best_candidate['Matched Skills'])} out of {len(job_skills_list)} required skills. Consider training for: {', '.join(best_candidate['Missing Skills'][:3])}")
        else:
            st.success(f"**Perfect match!** {best_candidate['Candidate']} has all required skills!")

if __name__ == "__main__":
    main()