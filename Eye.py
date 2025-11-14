#======================== IMPORT PACKAGES ===========================

import streamlit as st
import base64


st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:32px;">{"AI-Driven Advanced Techniques for Detecting Dry Eye Disease Using  Multi-Source Evidence: Case studies, Applications, Challenges, and Future Perspectives"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
    except FileNotFoundError:
        # If background image is not found, use a default gradient background
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        </style>
        """, unsafe_allow_html=True)
        return
    st.markdown(
    f"""
    <style>
    /* Import Google Fonts for professional typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Inter', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }}
    
    /* Professional Medical Theme Colors */
    :root {{
        --medical-blue: #0066CC;
        --medical-teal: #00A8A8;
        --medical-dark: #1A3A5F;
        --medical-light: #E8F4F8;
        --medical-white: #FFFFFF;
        --medical-accent: #4A90E2;
    }}
    
    /* Professional Headings with Medical Theme */
    h1, h2, h3, h4, h5, h6 {{
        background: linear-gradient(135deg, rgba(0, 102, 204, 0.95) 0%, rgba(0, 168, 168, 0.95) 100%) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 20px !important;
        padding: 20px 30px !important;
        margin: 15px 0 !important;
        box-shadow: 0 10px 40px rgba(0, 102, 204, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        color: #FFFFFF !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3) !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
        letter-spacing: 0.5px !important;
        text-align: center !important;
    }}
    
    /* Professional Text Elements */
    .stMarkdown, .stText, p, div, span, .stWrite {{
        color: #1A3A5F !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.6 !important;
    }}
    
    /* Professional Containers */
    .element-container, .stMarkdownContainer {{
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(0, 102, 204, 0.2) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        margin: 10px 0 !important;
        box-shadow: 0 8px 30px rgba(0, 102, 204, 0.15) !important;
    }}
    
    /* Professional Labels */
    .custom-label {{
        background: linear-gradient(135deg, #0066CC 0%, #00A8A8 100%) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 10px 15px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2) !important;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.3) !important;
        display: inline-block !important;
        margin-bottom: 8px !important;
    }}
    
    /* Professional Input Fields */
    .stTextInput > div > div > input {{
        background: rgba(255, 255, 255, 0.98) !important;
        border: 2px solid #0066CC !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
        font-size: 15px !important;
        font-family: 'Inter', sans-serif !important;
        color: #1A3A5F !important;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.1) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: #00A8A8 !important;
        box-shadow: 0 6px 20px rgba(0, 168, 168, 0.3) !important;
        outline: none !important;
    }}
    
    /* Professional Buttons - Medical Theme */
    .stButton > button {{
        background: linear-gradient(135deg, #0066CC 0%, #00A8A8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 6px 25px rgba(0, 102, 204, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2) !important;
        width: 100% !important;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, #0052A3 0%, #008B8B 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 35px rgba(0, 102, 204, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) !important;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.3) !important;
    }}
    
    /* Professional Success/Error Messages */
    .stSuccess {{
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.95) 0%, rgba(56, 142, 60, 0.95) 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 15px 20px !important;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3) !important;
        font-weight: 500 !important;
    }}
    
    .stError {{
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.95) 0%, rgba(198, 40, 40, 0.95) 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 15px 20px !important;
        box-shadow: 0 6px 20px rgba(244, 67, 54, 0.3) !important;
        font-weight: 500 !important;
    }}
    
    .stWarning {{
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.95) 0%, rgba(245, 124, 0, 0.95) 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 15px 20px !important;
        box-shadow: 0 6px 20px rgba(255, 152, 0, 0.3) !important;
        font-weight: 500 !important;
    }}
    
    .stInfo {{
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.95) 0%, rgba(25, 118, 210, 0.95) 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 15px 20px !important;
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.3) !important;
        font-weight: 500 !important;
    }}
    
    /* Hide Streamlit default elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Professional Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, #0066CC 0%, #00A8A8 100%);
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, #0052A3 0%, #008B8B 100%);
    }}
    
    /* Professional Image Containers */
    .stImage {{
        background: rgba(255, 255, 255, 0.95) !important;
        border: 3px solid #0066CC !important;
        border-radius: 15px !important;
        padding: 15px !important;
        box-shadow: 0 8px 30px rgba(0, 102, 204, 0.2) !important;
    }}
    
    /* Better spacing for main container */
    .main .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('2.jpg')   


# --------------------- REGISTER PAGE

st.markdown(f'<h1 style="color:#ef1ae8;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Register Here !!!"}</h1>', unsafe_allow_html=True)



import streamlit as st
import sqlite3
import re

# Function to create a database connection
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn

# Function to create a new user
def create_user(conn, user):
    sql = ''' INSERT INTO users(name, password, email, phone)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, user)
    conn.commit()
    return cur.lastrowid

# Function to check if a user already exists
def user_exists(conn, email):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email=?", (email,))
    if cur.fetchone():
        return True
    return False

# Function to validate email
def validate_email(email):
    pattern = r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
    return re.match(pattern, email)

# Function to validate phone number
def validate_phone(phone):
    pattern = r'^[6-9]\d{9}$'
    return re.match(pattern, phone)

# Main function
def main():
    # st.title("User Registration")

    # Create a database connection
    conn = create_connection("dbs.db")

    if conn is not None:
        # Create users table if it doesn't exist
        conn.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY,
                     name TEXT NOT NULL,
                     password TEXT NOT NULL,
                     email TEXT NOT NULL UNIQUE,
                     phone TEXT NOT NULL);''')

        # User input fields
        
        st.markdown(
            """
            <style>
            .custom-label {
                font-size: 13px; /* Change the font size */
                color: #000000;  /* Change the color */
                font-weight: bold; /* Optional: make text bold */
                display: inline-block; /* Make label inline with the input */
                margin-right: 10px; /* Adjust the space between label and input */
            }
            .custom-input {
                vertical-align: middle; /* Align input vertically with label */
            }
            </style>
            <label class="custom-label">Enter your name:</label>
            """,
            unsafe_allow_html=True
        )
        name = st.text_input("", label_visibility="hidden")
        

        # Create the text input field and password field
        # name = st.text_input("Your name")
        
        st.markdown(
            """
            <style>
            .custom-label {
                font-size: 13px; /* Change the font size */
                color: #000000;  /* Change the color */
                font-weight: bold; /* Optional: make text bold */
                display: inline-block; /* Make label inline with the input */
                margin-right: 10px; /* Adjust the space between label and input */
            }
            .custom-input {
                vertical-align: middle; /* Align input vertically with label */
            }
            </style>
            <label class="custom-label">Enter your Password:</label>
            """,
            unsafe_allow_html=True
        )
        
        password = st.text_input("", type="password", label_visibility="hidden")

        
        st.markdown(
            """
            <style>
            .custom-label {
                font-size: 13px; /* Change the font size */
                color: #000000;  /* Change the color */
                font-weight: bold; /* Optional: make text bold */
                display: inline-block; /* Make label inline with the input */
                margin-right: 10px; /* Adjust the space between label and input */
            }
            .custom-input {
                vertical-align: middle; /* Align input vertically with label */
            }
            </style>
            <label class="custom-label">Enter your Confirm Password:</label>
            """,
            unsafe_allow_html=True
        )
        
        confirm_password = st.text_input(" ", type="password", label_visibility="hidden")
        
        # ------

        st.markdown(
            """
            <style>
            .custom-label {
                font-size: 13px; /* Change the font size */
                color: #000000;  /* Change the color */
                font-weight: bold; /* Optional: make text bold */
                display: inline-block; /* Make label inline with the input */
                margin-right: 10px; /* Adjust the space between label and input */
            }
            .custom-input {
                vertical-align: middle; /* Align input vertically with label */
            }
            </style>
            <label class="custom-label">Enter your Email ID:</label>
            """,
            unsafe_allow_html=True
        )

        email = st.text_input("  ", label_visibility="hidden")
        
        
        st.markdown(
            """
            <style>
            .custom-label {
                font-size: 13px; /* Change the font size */
                color: #000000;  /* Change the color */
                font-weight: bold; /* Optional: make text bold */
                display: inline-block; /* Make label inline with the input */
                margin-right: 10px; /* Adjust the space between label and input */
            }
            .custom-input {
                vertical-align: middle; /* Align input vertically with label */
            }
            </style>
            <label class="custom-label">Enter your Phone Number:</label>
            """,
            unsafe_allow_html=True
        )
        
        
        phone = st.text_input("   ", label_visibility="hidden")

        col1, col2 = st.columns(2)

        with col1:
                
            aa = st.button("REGISTER")
            
            if aa:
                
                if password == confirm_password:
                    if not user_exists(conn, email):
                        if validate_email(email) and validate_phone(phone):
                            user = (name, password, email, phone)
                            create_user(conn, user)
                            st.success("User registered successfully!")
                        else:
                            st.error("Invalid email or phone number!")
                    else:
                        st.error("User with this email already exists!")
                else:
                    st.error("Passwords do not match!")
                
                conn.close()
                # st.success('Successfully Registered !!!')
            # else:
                
                # st.write('Registeration Failed !!!')     
        
        with col2:
                
            aa = st.button("LOGIN")
            
            if aa:
                import subprocess
                subprocess.run(['python','-m','streamlit','run','Login.py'])



  
if __name__ == '__main__':
    main()


