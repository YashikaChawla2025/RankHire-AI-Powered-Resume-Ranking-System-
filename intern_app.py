import streamlit as st
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Check if text was extracted
                    text += page_text
                else:
                    st.warning(f"No text found on page {page.page_number} of {file.name}.")
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Streamlit app
st.set_page_config(page_title="RankHire", page_icon=":guardsman:", layout="wide")
st.title("RankHire")


# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        color: #000000;  /* Black for text */
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent white */
        color: #000000;  /* Black for text */
    }
    .stButton>button {
        background-color: #6A0DAD;  /* Dark Purple */
        color: #FFFFFF;  /* White for text */
    }
    .stTextInput>div>input {
        background-color: #FFFFFF;  /* White */
        color: #333333;  /* Dark Gray for text */
    }
    .stTextArea>div>textarea {
        background-color: #FFFFFF;  /* White */
        color: #333333;  /* Dark Gray for text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation with buttons
st.sidebar.image('images\logo(1).png',width=250 )  # Update with your sidebar logo path
st.sidebar.title("Dashboard")
if st.sidebar.button("Login", key="login_button"):
    st.session_state["show_login"] = True
    st.session_state["show_main"] = False
    st.session_state["show_feedback"] = False
    st.session_state["show_history"] = False

if st.sidebar.button("Main Page", key="main_page_button"):
    st.session_state["show_main"] = True
    st.session_state["show_login"] = False
    st.session_state["show_feedback"] = False
    st.session_state["show_history"] = False

if st.sidebar.button("Feedback", key="feedback_button"):
    st.session_state["show_feedback"] = True
    st.session_state["show_login"] = False
    st.session_state["show_main"] = False
    st.session_state["show_history"] = False

if st.sidebar.button("Manage History", key="manage_history_button"):
    st.session_state["show_history"] = True
    st.session_state["show_login"] = False
    st.session_state["show_main"] = False
    st.session_state["show_feedback"] = False

# Login functionality
def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login", key="login_submit_button"):
        # Allow any username and password
        st.session_state["logged_in"] = True
        st.success("Login successful!")
        st.session_state["show_login"] = False

# Check if user is logged in
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "show_login" not in st.session_state:
    st.session_state["show_login"] = True  # Show login by default

if "show_history" not in st.session_state:
    st.session_state["show_history"] = False

if "show_feedback" not in st.session_state:
    st.session_state["show_feedback"] = False

if "show_main" not in st.session_state:
    st.session_state["show_main"] = True

if "history" not in st.session_state:
    st.session_state["history"] = []  # Initialize history

if "feedback_history" not in st.session_state:
    st.session_state["feedback_history"] = []  # Initialize feedback history

if st.session_state["show_login"] or not st.session_state["logged_in"]:
    login()
else:
    if st.session_state["show_history"]:
        # Manage History Page
        st.header("Manage History")
        
        # Display Job Description History
        st.subheader("Job Description History")
        if st.session_state["history"]:
            history_df = pd.DataFrame(st.session_state["history"], columns=["Job Description", "Resumes", "Scores"])
            st.write(history_df)
        else:
            st.write("No history available.")

        # Display Feedback History
        st.subheader("Feedback History")
        if st.session_state["feedback_history"]:
            feedback_df = pd.DataFrame(st.session_state["feedback_history"], columns=["Feedback"])
            st.write(feedback_df)
        else:
            st.write("No feedback available.")

        if st.button("Back to Main", key="back_to_main_from_history"):
            st.session_state["show_history"] = False
    elif st.session_state["show_feedback"]:
        # Feedback Page
        st.header("Feedback")
        feedback = st.text_area("Please provide your feedback here:")
        
        if st.button("Submit Feedback", key="submit_feedback_button"):
            if feedback:
                st.success("Thank you for your feedback!")
                st.session_state["feedback_history"].append(feedback)  # Store feedback in session state
            else:
                st.error("Please enter your feedback before submitting.")
        
        if st.button("Back to Main", key="back_to_main_from_feedback"):
            st.session_state["show_feedback"] = False
    elif st.session_state["show_main"]:
        # Job description input
        st.header("Job Description")
        job_description = st.text_area("Enter the job description")

        # File uploader
        st.header("Upload Resumes")
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if uploaded_files and job_description:
            resumes = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)  # Extract text from each PDF file
                resumes.append(text)

            # Rank resumes
            scores = rank_resumes(job_description, resumes)

            # Create a DataFrame to display results
            results = pd.DataFrame({
                "Resumes": [file.name for file in uploaded_files],
                "Score": scores
            })

            # Store the history
            st.session_state["history"].append((job_description, [file.name for file in uploaded_files], scores.tolist()))

            # Sort results by score
            results = results.sort_values(by="Score", ascending=False)
            st.write(results)

            # Display a pie chart of the scores
            fig, ax = plt.subplots()
            ax.pie(results["Score"], labels=results["Resumes"], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)
        else:
            st.warning("Please enter a job description and upload resumes to see the ranking.")