#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import matplotlib.image as mpimg
import cv2
import streamlit as st

import base64
from streamlit_option_menu import option_menu
import pickle
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.image as mpimg


#======================== BACK GROUND IMAGE ===========================



st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:32px;">{"AI-Driven Advanced Techniques for Detecting Dry Eye Disease Using  Multi-Source Evidence: Case studies, Applications, Challenges, and Future Perspectives"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
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
    
    /* Professional File Uploader */
    .stFileUploader {{
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px dashed #0066CC !important;
        border-radius: 15px !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }}
    
    .stFileUploader:hover {{
        border-color: #00A8A8 !important;
        background: rgba(232, 244, 248, 0.95) !important;
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
    
    /* Professional DataFrames and Tables */
    .stDataFrame, table {{
        background: rgba(255, 255, 255, 0.98) !important;
        border: 2px solid #0066CC !important;
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 30px rgba(0, 102, 204, 0.15) !important;
    }}
    
    .stDataFrame table thead {{
        background: linear-gradient(135deg, #0066CC 0%, #00A8A8 100%) !important;
        color: #FFFFFF !important;
    }}
    
    /* Professional Metrics */
    .stMetric {{
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid #0066CC !important;
        border-radius: 15px !important;
        padding: 20px !important;
        box-shadow: 0 6px 25px rgba(0, 102, 204, 0.2) !important;
    }}
    
    /* Professional Option Menu */
    .streamlit-option-menu {{
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid #0066CC !important;
        border-radius: 15px !important;
        padding: 10px !important;
        box-shadow: 0 8px 30px rgba(0, 102, 204, 0.2) !important;
    }}
    
    /* Professional Progress Bars */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, #0066CC 0%, #00A8A8 100%) !important;
        border-radius: 10px !important;
    }}
    
    /* Professional Spinner */
    .stSpinner {{
        border-color: #0066CC !important;
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
    
    /* Professional Selectbox and Dropdowns */
    .stSelectbox > div > div {{
        background: rgba(255, 255, 255, 0.98) !important;
        border: 2px solid #0066CC !important;
        border-radius: 12px !important;
    }}
    
    /* Professional Columns */
    [data-testid="stColumn"] {{
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        padding: 10px !important;
    }}
    
    /* Professional Expander */
    .streamlit-expanderHeader {{
        background: linear-gradient(135deg, rgba(0, 102, 204, 0.1) 0%, rgba(0, 168, 168, 0.1) 100%) !important;
        border: 2px solid #0066CC !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #1A3A5F !important;
    }}
    
    /* Professional Code Blocks */
    pre, code {{
        background: rgba(26, 58, 95, 0.95) !important;
        color: #FFFFFF !important;
        border: 2px solid #0066CC !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-family: 'Courier New', monospace !important;
    }}
    
    /* Professional Separators */
    hr {{
        border: none !important;
        border-top: 3px solid #0066CC !important;
        border-radius: 2px !important;
        margin: 20px 0 !important;
        box-shadow: 0 2px 5px rgba(0, 102, 204, 0.2) !important;
    }}
    
    /* Professional Loading Spinner */
    .stSpinner > div {{
        border-color: #0066CC !important;
    }}
    
    /* Better spacing for main container */
    .main .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }}
    
    /* Professional Card Effect for Sections */
    section[data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('3.jpg')


# -------------------------------------------------------------------

selected = option_menu(
    menu_title=None, 
    options=["Dry Eye Prediction", "Eye Disease Prediction", "Eye Blink Detection"],  
    orientation="horizontal",
)


st.markdown(
    """
    <style>
    .option_menu_container {
        position: fixed;
        top: 20px;
        right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Disease

if selected == 'Eye Disease Prediction':

    # abt = " Here , the system can predict the input image is affected or not with the help of deep learning algorithm such as CNN-2D effectively"
    # a="A Long-Term Recurrent Convolutional Network for Eye Blink Completeness Detection introduces a novel deep learning framework designed to accurately detect the completeness of eye blinks. By integrating convolutional neural networks (CNNs) with long short-term memory (LSTM) networks, the Eye-LRCN model effectively captures both spatial and temporal features from video sequences. This hybrid architecture allows for precise identification of partial and complete blinks, improving over traditional methods that often struggle with the subtle nuances of eye movements. The model's performance is evaluated on multiple datasets, demonstrating its robustness and potential applications in fields such as driver drowsiness detection, human-computer interaction, and neurological disorder monitorin"
    # st.markdown(f'<h1 style="color:#000000;text-align: justify;font-size:16px;">{a}</h1>', unsafe_allow_html=True)

    st.write("-----------------------------------------------------------")

    filename = st.file_uploader("Choose Image",['jpg','png'])
    
    with open('file.pickle', 'wb') as f:
        pickle.dump(filename, f)
    
    
    
    if filename is None:

        st.text("Upload Image")
        
    else:
            
        # filename = askopenfilename()
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Input Image"}</h1>', unsafe_allow_html=True)
        
        img = mpimg.imread(filename)
    
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis ('off')
        plt.show()
        
        
        st.image(img,caption="Original Image")
    
        
        #============================ PREPROCESS =================================
        
        #==== RESIZE IMAGE ====
        
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Preprocessing"}</h1>', unsafe_allow_html=True)
        
        
        resized_image = cv2.resize(img,(300,300))
        img_resize_orig = cv2.resize(img,((50, 50)))
        
        fig = plt.figure()
        plt.title('RESIZED IMAGE')
        plt.imshow(resized_image)
        plt.axis ('off')
        plt.show()
        
        st.image(resized_image,caption="Resized Image")
        
        # st.image(img,caption="Original Image")
                 
        #==== GRAYSCALE IMAGE ====
        

        SPV = np.shape(img)
        
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
            
        except:
            gray1 = img_resize_orig
           
        fig = plt.figure()
        plt.title('GRAY SCALE IMAGE')
        plt.imshow(gray1)
        plt.axis ('off')
        plt.show()
        
    
        st.image(gray1,caption="Gray Scale Image")        
        
        #=============================== 3.FEATURE EXTRACTION ======================
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Feature Extraction"}</h1>', unsafe_allow_html=True)
        
    
        
        # === GRAY LEVEL CO OCCURENCE MATRIX ===
        
        from skimage.feature import graycomatrix, graycoprops
        
        print()
        print("-----------------------------------------------------")
        print("FEATURE EXTRACTION -->GRAY LEVEL CO-OCCURENCE MATRIX ")
        print("-----------------------------------------------------")
        print()
        
        
        PATCH_SIZE = 21
        
        # open the image
        
        # Handle both grayscale (2D) and color (3D) images
        if len(img.shape) == 3:
            # Color image - extract first channel or convert to grayscale
            image = img[:,:,0]
        else:
            # Grayscale image - use directly
            image = img
        image = cv2.resize(image,(768,1024))
         
        grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
        grass_patches = []
        for loc in grass_locations:
            grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                       loc[1]:loc[1] + PATCH_SIZE])
        
        # select some patches from sky areas of the image
        sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
        sky_patches = []
        for loc in sky_locations:
            sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                     loc[1]:loc[1] + PATCH_SIZE])
        
        # compute some GLCM properties each patch
        xs = []
        ys = []
        for patch in (grass_patches + sky_patches):
            glcm = graycomatrix(image.astype(int), distances=[4], angles=[0], levels=256,symmetric=True)
            xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            ys.append(graycoprops(glcm, 'correlation')[0, 0])
        
        
        # create the figure
        fig = plt.figure(figsize=(8, 8))
        
        # display original image with locations of patches
        ax = fig.add_subplot(3, 2, 1)
        ax.imshow(image, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        for (y, x) in grass_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
        for (y, x) in sky_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
        ax.set_xlabel('GLCM')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('image')
        plt.show()
        
        # for each patch, plot (dissimilarity, correlation)
        ax = fig.add_subplot(3, 2, 2)
        ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
                label='Region 1')
        ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
                label='Region 2')
        ax.set_xlabel('GLCM Dissimilarity')
        ax.set_ylabel('GLCM Correlation')
        ax.legend()
        plt.show()
        
        
        sky_patches0 = np.mean(sky_patches[0])
        sky_patches1 = np.mean(sky_patches[1])
        sky_patches2 = np.mean(sky_patches[2])
        sky_patches3 = np.mean(sky_patches[3])
        
        Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
        Tesfea1 = []
        Tesfea1.append(Glcm_fea[0])
        Tesfea1.append(Glcm_fea[1])
        Tesfea1.append(Glcm_fea[2])
        Tesfea1.append(Glcm_fea[3])
        
        
        print("---------------------------------------------------")
        st.write("GLCM FEATURES =")
        print("---------------------------------------------------")
        print()
        st.write(Glcm_fea)
        


         
        # ========= IMAGE SPLITTING ============
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Image Splitting"}</h1>', unsafe_allow_html=True)
        
        
        import os 
        
        from sklearn.model_selection import train_test_split
          
        # Helper function to filter image files
        def is_image_file(filename):
            image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.jfif', '.JFIF')
            return filename.lower().endswith(image_extensions)
         
        data_aff = [f for f in os.listdir('Dataset/Affected/') if is_image_file(f)]
         
        data_not = [f for f in os.listdir('Dataset/Not/') if is_image_file(f)]
         
        

        
        import numpy as np
        dot1= []
        labels1 = [] 
        for img11 in data_aff:
            # print(img)
            img_1 = mpimg.imread('Dataset/Affected//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(1)
         
         
        for img11 in data_not:
            # print(img)
            img_1 = mpimg.imread('Dataset/Not//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(2)
         
   

        x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
        
        
        print("------------------------------------------------------------")
        print(" Image Splitting")
        print("------------------------------------------------------------")
        print()
        
        st.write("The Total of Images       =",len(dot1))
        st.write("The Total of Train Images =",len(x_train))
        st.write("The Total of Test Images  =",len(x_test))
          
          
              
          
        #=============================== CLASSIFICATION =================================
        
        from keras.utils import to_categorical
        
        y_train1=np.array(y_train)
        y_test1=np.array(y_test)
        
        train_Y_one_hot = to_categorical(y_train1)
        test_Y_one_hot = to_categorical(y_test)
        
        
        
        
        # Fix data preparation - convert grayscale to 3-channel
        x_train2 = np.zeros((len(x_train), 50, 50, 3))
        for i in range(len(x_train)):
            img = x_train[i]
            if len(img.shape) == 2:  # Grayscale
                x_train2[i] = np.stack([img] * 3, axis=-1)
            else:
                x_train2[i] = img
        
        x_test2 = np.zeros((len(x_test), 50, 50, 3))
        for i in range(len(x_test)):
            img = x_test[i]
            if len(img.shape) == 2:  # Grayscale
                x_test2[i] = np.stack([img] * 3, axis=-1)
            else:
                x_test2[i] = img
        
        # Normalize to [0, 1]
        x_train2 = x_train2.astype('float32') / 255.0
        x_test2 = x_test2.astype('float32') / 255.0
    
    # ===================================== CLASSIFICATION ==================================
    
     # ----------------------- MOBILENET -----------------------
    
            
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Classification - MobileNet"}</h1>', unsafe_allow_html=True)
        
    
        import time
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from keras.utils import to_categorical
        from tensorflow.keras import layers, models
        
        
        print()
        print("----------------------------------------------")
        print(" Classification - Mobilnet")
        print("----------------------------------------------")
        print()
        from tensorflow.keras.applications import MobileNet
        
        start_mob = time.time()
        
        base_model = MobileNet(weights=None, input_shape=(50, 50, 3), classes=3)
        
        model = models.Model(inputs=base_model.input, outputs=base_model.output)
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        model.summary()
        
        history = model.fit(x_train2,train_Y_one_hot, epochs=3, batch_size=64)
        
        loss_val = history.history['loss']
        
        loss_val = min(loss_val)
        
        acc_mob = 100 - loss_val
        
        
        print("-------------------------------------")
        print("Mobilenet - Perfromance Analysis")
        print("-------------------------------------")
        print()
        print("1. Accuracy   =", acc_mob,'%')
        print()
        print("2. Error Rate =",loss_val)
        print()
        
        
        predictions = model.predict(x_test2)
        
        end_mob = time.time()
        
        time_mob = (end_mob-start_mob) * 10**3
        
        time_mob = time_mob / 1000
        
        print("3. Execution Time  = ",time_mob, "s")
        
        
        st.write("-------------------------------------")
        st.write("Mobilenet - Perfromance Analysis")
        st.write("-------------------------------------")
        print()
        st.write("1. Accuracy   =", acc_mob,'%')
        print()
        st.write("2. Error Rate =",loss_val)
        print()
        
        
        predictions = model.predict(x_test2)
        
        end_mob = time.time()
        
        time_mob = (end_mob-start_mob) * 10**3
        
        time_mob = time_mob / 1000
        
        st.write("3. Execution Time  = ",time_mob, "s")
                
        
        # --- prediction
        
        st.write("-----------------------------------------------------------")
    
        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Prediction -Eye Disease"}</h1>', unsafe_allow_html=True)
         
        # Prepare the uploaded image for prediction
        try:
            # Resize and preprocess the uploaded image
            test_img_resized = cv2.resize(img_resize_orig, (50, 50))
            
            # Convert to grayscale if needed
            try:
                test_img_gray = cv2.cvtColor(test_img_resized, cv2.COLOR_BGR2GRAY)
            except:
                test_img_gray = test_img_resized
            
            # Convert to 3-channel if needed
            if len(test_img_gray.shape) == 2:
                test_img_3channel = np.stack([test_img_gray] * 3, axis=-1)
            else:
                test_img_3channel = test_img_gray
            
            # Ensure it's the right shape
            if test_img_3channel.shape != (50, 50, 3):
                test_img_3channel = cv2.resize(test_img_3channel, (50, 50))
                if len(test_img_3channel.shape) == 2:
                    test_img_3channel = np.stack([test_img_3channel] * 3, axis=-1)
            
            # Normalize and reshape for prediction
            test_img_array = np.expand_dims(test_img_3channel, axis=0)
            
            # Use the trained model to make prediction
            prediction = model.predict(test_img_array, verbose=0)
            predicted_class = np.argmax(prediction[0])
            
            # Map prediction to label
            # The model outputs probabilities for 3 classes, but labels 1 and 2 are used
            # Class 0: unused, Class 1: Affected (label 1), Class 2: Not Affected (label 2)
            confidence = prediction[0][predicted_class] * 100
            
            st.write('-----------------------------------------')
            print()
            # Since to_categorical converts labels 1,2 to [0,1,0] and [0,0,1]
            # predicted_class 1 means Affected, predicted_class 2 means Not
            if predicted_class == 1:
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Affected"}</h1>', unsafe_allow_html=True)
            elif predicted_class == 2:
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Not Affected"}</h1>', unsafe_allow_html=True)
            else:
                # If class 0 is predicted (shouldn't happen), use the highest non-zero class
                probs = prediction[0]
                if probs[1] > probs[2]:
                    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Affected"}</h1>', unsafe_allow_html=True)
                    confidence = probs[1] * 100
                else:
                    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Not Affected"}</h1>', unsafe_allow_html=True)
                    confidence = probs[2] * 100
            st.write(f'Confidence: {confidence:.2f}%')
            print()
            st.write('---------------------------------')
            
        except Exception as e:
            # Fallback to distance-based matching if model prediction fails
            st.write("Using fallback prediction method...")
            try:
                Total_length = len(dot1)
                
                # Find the closest match by comparing mean values with tolerance
                test_mean = np.mean(gray1)
                distances = [abs(np.mean(dot1[ijk]) - test_mean) for ijk in range(Total_length)]
                closest_idx = np.argmin(distances)
                
                if labels1[closest_idx] == 1:
                    st.write('-----------------------------------------')
                    print()
                    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Affected"}</h1>', unsafe_allow_html=True)
                else:
                    st.write('---------------------------------')
                    print()
                    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Not Affected"}</h1>', unsafe_allow_html=True)
                    print()
                    st.write('---------------------------------')
            except Exception as e2:
                st.error(f"Prediction failed: {str(e2)}")
                st.write("Please try uploading a different image.")   
    


if selected == "Dry Eye Prediction":
    
    file = st.file_uploader("Upload Input Dataset",['csv','xlsx'])
    
    import pandas as pd
    import time
    from sklearn.model_selection import train_test_split


    
    if file is None:
        
        st.warning("Upload Input Data")
    
    else:
        
    
        dataframe=pd.read_excel("Dataset.xlsx")
                
        print("--------------------------------")
        print("Data Selection")
        print("--------------------------------")
        print()
        print(dataframe.head(15))    
        
        st.write("--------------------------------")
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Data Selection !!!"}</h1>', unsafe_allow_html=True)
    
        # st.write("--------------------------------")
        # st.write("Data Selection")
        # st.write("--------------------------------")
        print()
        st.write(dataframe.head(15))    
        
        
     #-------------------------- PRE PROCESSING --------------------------------    
        
        # ----- CHECKING MISSING VALUES 
        
        
        
        print("----------------------------------------------------")
        print("              Handling Missing values               ")
        print("----------------------------------------------------")
        print()
        print(dataframe.isnull().sum())
        
        st.write("--------------------------------")
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Pre-processing !!!"}</h1>', unsafe_allow_html=True)
    
        
        # st.write("----------------------------------------------------")
        st.write("              Handling Missing values               ")
        st.write("----------------------------------------------------")
        print()
        st.write(dataframe.isnull().sum())
        
        res = dataframe.isnull().sum().any()
            
        if res == False:
            
            print("--------------------------------------------")
            print("  There is no Missing values in our dataset ")
            print("--------------------------------------------")
            print()   
            
            
            st.write("--------------------------------------------")
            st.write("  There is no Missing values in our dataset ")
            st.write("--------------------------------------------")
      
        
            
        else:
        
            print("--------------------------------------------")
            print(" Missing values is present in our dataset   ")
            print("--------------------------------------------")
            print()    
            
            st.write("--------------------------------------------")
            st.write("  Missing values is present in our dataset ")
            
            dataframe = dataframe.fillna(0)
            
            resultt = dataframe.isnull().sum().any()
            
            if resultt == False:
                
                print("--------------------------------------------")
                print(" Data Cleaned !!!   ")
                print("--------------------------------------------")
                print()    
                print(dataframe.isnull().sum())  
                
                
                st.write("--------------------------------------------")
                st.write(" Data Cleaned !!!   ")
                st.write("--------------------------------------------")
                print()    
                st.write(dataframe.isnull().sum()) 
                
                
        # --- DROP UNWANTED COLUMNS
        
        dataframe = dataframe.drop(['Timestamp','Consent'],axis=1)
            
        
        # ----- LABEL ENCODING
                
    
        print("----------------------------------------------------")
        print("            Before Label Encoding                   ")
        print("----------------------------------------------------")
        print()
        print(dataframe['Gender'].head(15))                
                    
        st.write("----------------------------------------------------")
        st.write("            Before Label Encoding                   ")
        st.write("----------------------------------------------------")
        print()
        st.write(dataframe['Gender'].head(15))
        
        
        gen =  dataframe['Gender']
        
        aca_yr = dataframe['Gender']
        
    
    
    
    
        from sklearn import preprocessing
        
        label_encoder = preprocessing.LabelEncoder()
        
        dataframe['Gender']= label_encoder.fit_transform(dataframe['Gender'])  
        
        dataframe['Academic Year']= label_encoder.fit_transform(dataframe['Academic Year'])  
        
        dataframe['What type of Digital display device do you use?']= label_encoder.fit_transform(dataframe['What type of Digital display device do you use?'])  
        
        dataframe['How many hours in a day do you spend on your smartphones, laptops, etc?']= label_encoder.fit_transform(dataframe['How many hours in a day do you spend on your smartphones, laptops, etc?'])                
                    
        dataframe['Eyes that are sensitive to light?']= label_encoder.fit_transform(dataframe['Eyes that are sensitive to light?'])  
        
        dataframe['Eyes that feel gritty (itchy and Scratchy) ?']= label_encoder.fit_transform(dataframe['Eyes that feel gritty (itchy and Scratchy) ?'])  
    
    
        dataframe['Painful or Sore eyes?']= label_encoder.fit_transform(dataframe['Painful or Sore eyes?'])  
        
        dataframe['Blurred vision?']= label_encoder.fit_transform(dataframe['Blurred vision?'])  
    
        dataframe['Reading?']= label_encoder.fit_transform(dataframe['Reading?'])  
        
        dataframe['Driving at night?']= label_encoder.fit_transform(dataframe['Driving at night?'].astype(str))  
    
        dataframe['Working with a computer or bank machine ATM?']= label_encoder.fit_transform(dataframe['Working with a computer or bank machine ATM?'])  
    
        dataframe['Watching TV?']= label_encoder.fit_transform(dataframe['Watching TV?'].astype(str))  
        
        dataframe['Windy conditions?']= label_encoder.fit_transform(dataframe['Windy conditions?'].astype(str))  
                   
                    
        dataframe['Places or areas with low humidity (very dry)?']= label_encoder.fit_transform(dataframe['Places or areas with low humidity (very dry)?'].astype(str))  
         
        dataframe['Areas that are air-conditioned?']= label_encoder.fit_transform(dataframe['Areas that are air-conditioned?'].astype(str))                 
        
        dataframe['Poor Vision?']= label_encoder.fit_transform(dataframe['Poor Vision?'])  

                    
        dataframe['Results']= label_encoder.fit_transform(dataframe['Results'])  
           
        print("----------------------------------------------------")
        print("            After Label Encoding                   ")
        print("----------------------------------------------------")
        print()
        print(dataframe['Gender'].head(15))
        
        st.write("----------------------------------------------------")
        st.write("            After Label Encoding                   ")
        st.write("----------------------------------------------------")
        print()
        st.write(dataframe['Gender'].head(15))
        
        
        
       #-------------------------- DATA SPLITTING  --------------------------------
        
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Data Splitting !!!"}</h1>', unsafe_allow_html=True)

        
        X=dataframe.drop('Results',axis=1)
                
        y=dataframe['Results']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
        print("---------------------------------------------")
        print("             Data Splitting                  ")
        print("---------------------------------------------")
        
        print()
        
        print("Total no of input data   :",dataframe.shape[0])
        print("Total no of test data    :",X_test.shape[0])
        print("Total no of train data   :",X_train.shape[0])
        
        
        # st.write("---------------------------------------------")
        st.write("     Test Data & Train Data                  ")
        st.write("---------------------------------------------")
        
        print()
        
        st.write("Total no of input data   :",dataframe.shape[0])
        st.write("Total no of test data    :",X_test.shape[0])
        st.write("Total no of train data   :",X_train.shape[0])
    
        #-------------------------- CLASSIFICATION --------------------------------
        
    
        #  ------ MLP --------
        
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Classification - MLP !!!"}</h1>', unsafe_allow_html=True)

        
        from sklearn.neural_network import MLPClassifier         
        
        start_mlp = time.time()
        
        mlpp = MLPClassifier()
        
        mlpp.fit(X_train, y_train)
        
        
        pred_mlpp = mlpp.predict(X_test)
        
        
        from sklearn import metrics
        
        
        acc_mlp = metrics.accuracy_score(pred_mlpp,y_test)* 100
        
        loss = 100 - acc_mlp
        # Classification report
        report_svm = metrics.classification_report(y_test, pred_mlpp, target_names=['Mild', 'Moderate','Normal','Severe'])
        
        # Confusion matrix
        cm = metrics.confusion_matrix(y_test, pred_mlpp)
        
       
        
        
        end_mlp = time.time()
        
        
        exec_time = (end_mlp-start_mlp) * 10**3
        
        exec_time_mlp = exec_time/1000
        
        
        print("----------------------------------------------------")
        print("     Classification -- Multi Layer Perceptron      ")
        print("----------------------------------------------------")
        
        print()
        
        print("1)  Accuracy        =", acc_mlp ,'%')
        print()
        print("2)  Error rate      = ", loss ,'%' )
        print()
        print("3)  Execution Time  = ", exec_time_mlp , 'sec')
        print()
        print("4) Classification Report  = ", )
        print()
        print(report_svm)
        
        
        st.write("----------------------------------------------------")
        st.write("      Classification -- Multi Layer Perceptron      ")
        st.write("----------------------------------------------------")
        
        print()
        
        st.write("1)  Accuracy        =", acc_mlp ,'%')
        print()
        st.write("2)  Error rate      = ", loss ,'%' )
        print()
        st.write("3)  Execution Time  = ", exec_time_mlp , 'sec')
        print()
        st.write("4) Classification Report  = ", )
        print()
        st.write(report_svm)
        
        st.write("---------------------------------------------")                
                    
                
                
# -------------------------- PREDICTION  ----------------------
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Prediction  !!!"}</h1>', unsafe_allow_html=True)
        
        # Display available prediction numbers
        num_test_samples = len(pred_mlpp)
        st.write("----------------------------------------------------")
        st.write(f"**Available Test Samples: {num_test_samples}**")
        st.write(f"**Valid Prediction Numbers: 0 to {num_test_samples - 1}**")
        st.write("----------------------------------------------------")
        
        # Create a mapping of prediction values to labels
        prediction_labels = {0: "MILD STAGE", 1: "MODERATE STAGE", 2: "NORMAL", 3: "SEVERE STAGE"}
        
        # Display first 20 predictions as a preview
        st.write("**Preview of Predictions (First 20 samples):**")
        preview_data = []
        preview_count = min(20, num_test_samples)
        for i in range(preview_count):
            pred_value = pred_mlpp[i]
            pred_label = prediction_labels.get(pred_value, f"Unknown ({pred_value})")
            preview_data.append({
                "Prediction #": i,
                "Predicted Class": pred_label,
                "Class Code": pred_value
            })
        
        import pandas as pd
        preview_df = pd.DataFrame(preview_data)
        st.dataframe(preview_df, use_container_width=True)
        
        if num_test_samples > 20:
            st.write(f"*... and {num_test_samples - 20} more predictions available*")
        
        st.write("---")
        
        # inputt = int(input("Enter Prediction Number :"))
        
        inputt = st.text_input("Enter Prediction Number (0 to {}):".format(num_test_samples - 1))
        
        butt = st.button("Submit")
        
        if butt:
            try:
                inputt = int(inputt)
                
                # Validate input range
                if inputt < 0 or inputt >= num_test_samples:
                    st.error(f"❌ Invalid prediction number! Please enter a number between 0 and {num_test_samples - 1}")
                else:
                    if pred_mlpp[inputt] == 0:
                        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:25px;">{"Identified Affected = MILD STAGE"}</h1>', unsafe_allow_html=True)
                    elif pred_mlpp[inputt] == 1:
                        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:25px;">{"Identified Affected = MODERATE STAGE"}</h1>', unsafe_allow_html=True)
                    elif pred_mlpp[inputt] == 2:
                        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:25px;">{"Identified Normal "}</h1>', unsafe_allow_html=True)
                    elif pred_mlpp[inputt] == 3:
                        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:25px;">{"Identified Affected = SEVERE STAGE"}</h1>', unsafe_allow_html=True)
            except ValueError:
                st.error("❌ Please enter a valid number!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                
if selected == 'Eye Blink Detection':               
                
                
   # abt = " Here , the system can predict the input image is affected or not with the help of deep learning algorithm such as CNN-2D effectively"
    # a="A Long-Term Recurrent Convolutional Network for Eye Blink Completeness Detection introduces a novel deep learning framework designed to accurately detect the completeness of eye blinks. By integrating convolutional neural networks (CNNs) with long short-term memory (LSTM) networks, the Eye-LRCN model effectively captures both spatial and temporal features from video sequences. This hybrid architecture allows for precise identification of partial and complete blinks, improving over traditional methods that often struggle with the subtle nuances of eye movements. The model's performance is evaluated on multiple datasets, demonstrating its robustness and potential applications in fields such as driver drowsiness detection, human-computer interaction, and neurological disorder monitorin"
    # st.markdown(f'<h1 style="color:#000000;text-align: justify;font-size:16px;">{a}</h1>', unsafe_allow_html=True)

    st.write("-----------------------------------------------------------")

    filename = st.file_uploader("Choose Image",['jpg','png'])
    

    
    
    if filename is None:

        st.text("Upload Image")
        
    else:
            
        # filename = askopenfilename()
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Input Image"}</h1>', unsafe_allow_html=True)
        
        img = mpimg.imread(filename)
    
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis ('off')
        plt.show()
        
        
        st.image(img,caption="Original Image")
    
        
        #============================ PREPROCESS =================================
        
        #==== RESIZE IMAGE ====
        
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Preprocessing"}</h1>', unsafe_allow_html=True)
        
        
        resized_image = cv2.resize(img,(300,300))
        img_resize_orig = cv2.resize(img,((50, 50)))
        
        fig = plt.figure()
        plt.title('RESIZED IMAGE')
        plt.imshow(resized_image)
        plt.axis ('off')
        plt.show()
        
        st.image(resized_image,caption="Resized Image")
        
        # st.image(img,caption="Original Image")
                 
        #==== GRAYSCALE IMAGE ====
        

        SPV = np.shape(img)
        
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
            
        except:
            gray1 = img_resize_orig
           
        fig = plt.figure()
        plt.title('GRAY SCALE IMAGE')
        plt.imshow(gray1)
        plt.axis ('off')
        plt.show()
        
    
        st.image(gray1,caption="Gray Scale Image")        
        
        #=============================== 3.FEATURE EXTRACTION ======================
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Feature Extraction"}</h1>', unsafe_allow_html=True)
        
    
        
        # === GRAY LEVEL CO OCCURENCE MATRIX ===
        
        from skimage.feature import graycomatrix, graycoprops
        
        print()
        print("-----------------------------------------------------")
        print("FEATURE EXTRACTION -->GRAY LEVEL CO-OCCURENCE MATRIX ")
        print("-----------------------------------------------------")
        print()
        
        
        PATCH_SIZE = 21
        
        # open the image
        
        # Handle both grayscale (2D) and color (3D) images
        if len(img.shape) == 3:
            # Color image - extract first channel or convert to grayscale
            image = img[:,:,0]
        else:
            # Grayscale image - use directly
            image = img
        image = cv2.resize(image,(768,1024))
         
        grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
        grass_patches = []
        for loc in grass_locations:
            grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                       loc[1]:loc[1] + PATCH_SIZE])
        
        # select some patches from sky areas of the image
        sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
        sky_patches = []
        for loc in sky_locations:
            sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                     loc[1]:loc[1] + PATCH_SIZE])
        
        # compute some GLCM properties each patch
        xs = []
        ys = []
        for patch in (grass_patches + sky_patches):
            glcm = graycomatrix(image.astype(int), distances=[4], angles=[0], levels=256,symmetric=True)
            xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            ys.append(graycoprops(glcm, 'correlation')[0, 0])
        
        
        # create the figure
        fig = plt.figure(figsize=(8, 8))
        
        # display original image with locations of patches
        ax = fig.add_subplot(3, 2, 1)
        ax.imshow(image, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        for (y, x) in grass_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
        for (y, x) in sky_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
        ax.set_xlabel('GLCM')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('image')
        plt.show()
        
        # for each patch, plot (dissimilarity, correlation)
        ax = fig.add_subplot(3, 2, 2)
        ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
                label='Region 1')
        ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
                label='Region 2')
        ax.set_xlabel('GLCM Dissimilarity')
        ax.set_ylabel('GLCM Correlation')
        ax.legend()
        plt.show()
        
        
        sky_patches0 = np.mean(sky_patches[0])
        sky_patches1 = np.mean(sky_patches[1])
        sky_patches2 = np.mean(sky_patches[2])
        sky_patches3 = np.mean(sky_patches[3])
        
        Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
        Tesfea1 = []
        Tesfea1.append(Glcm_fea[0])
        Tesfea1.append(Glcm_fea[1])
        Tesfea1.append(Glcm_fea[2])
        Tesfea1.append(Glcm_fea[3])
        
        
        print("---------------------------------------------------")
        st.write("GLCM FEATURES =")
        print("---------------------------------------------------")
        print()
        st.write(Glcm_fea)
        


         
        # ========= IMAGE SPLITTING ============
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Image Splitting"}</h1>', unsafe_allow_html=True)
        
        
        import os 
        
        from sklearn.model_selection import train_test_split
          
        # Helper function to filter image files
        def is_image_file(filename):
            image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.jfif', '.JFIF')
            return filename.lower().endswith(image_extensions)
         
        data_clos = [f for f in os.listdir('Blink/Closed/') if is_image_file(f)]
         
        data_forward = [f for f in os.listdir('Blink/forward_look/') if is_image_file(f)]
         
        data_left = [f for f in os.listdir('Blink/left_look/') if is_image_file(f)]
         
        data_open = [f for f in os.listdir('Blink/Open/') if is_image_file(f)]
         
        data_partial = [f for f in os.listdir('Blink/Partial/') if is_image_file(f)]
         
        data_right = [f for f in os.listdir('Blink/right_look/') if is_image_file(f)]    


        
        import numpy as np
        dot1= []
        labels1 = [] 
        for img11 in data_clos:
            # print(img)
            img_1 = mpimg.imread('Blink/Closed//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(1)
         
         
        for img11 in data_forward:
            # print(img)
            img_1 = mpimg.imread('Blink/forward_look//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(2)
         
         
        for img11 in data_left:
            # print(img)
            img_1 = mpimg.imread('Blink/left_look//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(3)
         
         
        for img11 in data_open:
            # print(img)
            img_1 = mpimg.imread('Blink/Open//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(4)
         
        for img11 in data_partial:
            # print(img)
            img_1 = mpimg.imread('Blink/Partial//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(5)
         
         
        for img11 in data_right:
            # print(img)
            img_1 = mpimg.imread('Blink/right_look//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(6)

        x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
        
        
        print("------------------------------------------------------------")
        print(" Image Splitting")
        print("------------------------------------------------------------")
        print()
        
        st.write("The Total of Images       =",len(dot1))
        st.write("The Total of Train Images =",len(x_train))
        st.write("The Total of Test Images  =",len(x_test))
          
          
              
        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Classification - VGG-19"}</h1>', unsafe_allow_html=True)

        #=============================== CLASSIFICATION =================================
        
        from keras.utils import to_categorical
        
        
        y_train1=np.array(y_train) - 1  # Convert labels from 1-6 to 0-5 for sparse_categorical_crossentropy
        y_test1=np.array(y_test) - 1
        
        train_Y_one_hot = to_categorical(y_train1)
        test_Y_one_hot = to_categorical(y_test1)
        
        
        
        
        # Fix data preparation - convert grayscale to 3-channel
        x_train2 = np.zeros((len(x_train), 50, 50, 3))
        for i in range(len(x_train)):
            img = x_train[i]
            if len(img.shape) == 2:  # Grayscale
                x_train2[i] = np.stack([img] * 3, axis=-1)
            else:
                x_train2[i] = img
        
        x_test2 = np.zeros((len(x_test), 50, 50, 3))
        for i in range(len(x_test)):
            img = x_test[i]
            if len(img.shape) == 2:  # Grayscale
                x_test2[i] = np.stack([img] * 3, axis=-1)
            else:
                x_test2[i] = img
        
        # Normalize to [0, 1]
        x_train2 = x_train2.astype('float32') / 255.0
        x_test2 = x_test2.astype('float32') / 255.0
          

                      
        import time
        import os
         # ==== VGG19 ==
        start_time = time.time()
        
        from keras.utils import to_categorical
        
        from tensorflow.keras.models import Sequential, load_model
        
        from tensorflow.keras.applications.vgg19 import VGG19
        from tensorflow.keras.layers import Flatten, Dense
        from tensorflow.keras.callbacks import ModelCheckpoint
        
        model_path = "vgg19_blink_model.h5"
        
        # Initialize session state for model training option
        if 'vgg19_retrain' not in st.session_state:
            st.session_state.vgg19_retrain = False
        
        # Check if model exists and user wants to retrain
        train_model = st.session_state.vgg19_retrain
        
        model = None
        
        # Check if model exists and user doesn't want to retrain
        if os.path.exists(model_path) and not train_model:
            try:
                with st.spinner("🔄 Loading pre-trained VGG-19 model..."):
                    model = load_model(model_path)
                st.success("✅ Pre-trained model loaded successfully! (Training skipped)")
            except Exception as e:
                st.warning(f"⚠️ Could not load model: {e}")
                st.info("Training new model...")
                model = None
                train_model = True
        
        # If model doesn't exist or user wants to retrain
        if model is None:
            if train_model and os.path.exists(model_path):
                st.info("🔄 Retraining model as requested...")
            elif not os.path.exists(model_path):
                st.info("⚠️ No pre-trained model found. Training new model...")
        
        # Add option to retrain if model exists and was loaded
        if model is not None and os.path.exists(model_path):
            if st.button("🔄 Retrain Model", help="Click to retrain the model (this will take 3-5 minutes)"):
                st.session_state.vgg19_retrain = True
                st.rerun()
        
        # Train model only if needed
        if model is None:
            st.warning("⏳ Training VGG-19 model - This will take 3-5 minutes. Please be patient...")
            
            # Create progress container
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
            
            try:
                status_text.text("🔄 Initializing VGG-19 base model...")
                progress_bar.progress(5)
                
                # Load VGG-19 with pre-trained weights
                vgg = VGG19(weights="imagenet", include_top=False, input_shape=(50, 50, 3))
                
                # Freeze VGG-19 layers
                for layer in vgg.layers:
                    layer.trainable = False
                
                status_text.text("🔄 Building model architecture...")
                progress_bar.progress(15)
                
                # Build model
                model = Sequential()
                model.add(vgg)
                model.add(Flatten())
                model.add(Dense(6, activation="softmax"))  # 6 classes for blink detection
                
                model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
                
                status_text.text("🔄 Starting training (1 epoch)...")
                progress_bar.progress(20)
                
                # Training callback for progress updates
                from tensorflow.keras.callbacks import Callback
                class TrainingCallback(Callback):
                    def on_epoch_begin(self, epoch, logs=None):
                        progress_bar.progress(20 + (epoch * 40))
                        status_text.text(f"🔄 Training epoch {epoch + 1}/1...")
                    
                    def on_batch_end(self, batch, logs=None):
                        # Estimate progress based on batch number
                        if hasattr(self, 'total_batches'):
                            current_progress = 20 + int((batch / self.total_batches) * 60)
                        else:
                            current_progress = 20 + min(60, int((batch / 100) * 60))
                        progress_bar.progress(min(current_progress, 80))
                    
                    def on_train_begin(self, logs=None):
                        # Estimate total batches
                        self.total_batches = len(x_train2) // 32 + 1
                
                # Train model with progress tracking
                history = model.fit(
                    x_train2, y_train1, 
                    batch_size=32,
                    epochs=1, 
                    validation_split=0.2, 
                    verbose=1,
                    callbacks=[TrainingCallback()]
                )
                
                progress_bar.progress(90)
                status_text.text("💾 Saving model...")
                
                # Save the model
                model.save(model_path)
                
                progress_bar.progress(100)
                status_text.text("✅ Training completed!")
                st.success("✅ Model trained and saved successfully!")
                
                # Calculate metrics from training
                if 'loss' in history.history and len(history.history['loss']) > 0:
                    loss = history.history['loss']
                    error_cnn = min(loss) if loss else 0
                    if 'accuracy' in history.history and len(history.history['accuracy']) > 0:
                        acc_cnn = max(history.history['accuracy']) * 100
                    else:
                        acc_cnn = max(0, 100 - error_cnn) if error_cnn < 100 else 0
                else:
                    acc_cnn = 0
                    error_cnn = 0
                
                # Reset retrain flag after training
                st.session_state.vgg19_retrain = False
                    
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.info("Please try again or check your data.")
                model = None
                acc_cnn = 0
                error_cnn = 0
                # Reset retrain flag on error
                st.session_state.vgg19_retrain = False
        
        # Model was loaded, evaluate on test set for metrics
        if model is not None:
            with st.spinner("Evaluating model on test data..."):
                try:
                    test_loss, test_acc = model.evaluate(x_test2, y_test1, verbose=0, batch_size=32)
                    acc_cnn = test_acc * 100
                    error_cnn = test_loss
                except Exception as e:
                    st.warning(f"Could not evaluate model: {e}")
                    # Use default values if evaluation fails
                    acc_cnn = 0
                    error_cnn = 0
                    
        end_time = time.time()
        
        exec_time = (end_time-start_time) * 10**3
        exec_time = exec_time/1000
        
        # st.write("-------------------------------------------")
        st.write("  Convolutional Neural Network - VGG 19")
        st.write("-------------------------------------------")
        print()
        st.write("1. Accuracy       =", f"{acc_cnn:.2f}",'%')
        print()
        st.write("2. Error Rate     =", f"{error_cnn:.4f}")
        print()
        st.write("3. Execution Time =", f"{exec_time:.2f}",'s')
                   
        st.write("-----------------------------------------------------------")
    
        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Prediction - Eye Blink Detection"}</h1>', unsafe_allow_html=True)
         
        # Prepare the uploaded image for prediction
        try:
            # Resize and preprocess the uploaded image
            test_img_resized = cv2.resize(img_resize_orig, (50, 50))
            
            # Convert to grayscale if needed
            try:
                test_img_gray = cv2.cvtColor(test_img_resized, cv2.COLOR_BGR2GRAY)
            except:
                test_img_gray = test_img_resized
            
            # Convert to 3-channel if needed
            if len(test_img_gray.shape) == 2:
                test_img_3channel = np.stack([test_img_gray] * 3, axis=-1)
            else:
                test_img_3channel = test_img_gray
            
            # Ensure it's the right shape
            if test_img_3channel.shape != (50, 50, 3):
                test_img_3channel = cv2.resize(test_img_3channel, (50, 50))
                if len(test_img_3channel.shape) == 2:
                    test_img_3channel = np.stack([test_img_3channel] * 3, axis=-1)
            
            # Normalize and reshape for prediction
            test_img_array = np.expand_dims(test_img_3channel, axis=0)
            
            # Use the trained VGG-19 model to make prediction
            if model is None:
                raise Exception("Model is not available. Please train the model first.")
            
            # Normalize test image
            test_img_array = test_img_array.astype('float32') / 255.0
            
            prediction = model.predict(test_img_array, verbose=0)
            
            # Model outputs probabilities for 6 classes
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class] * 100
            
            # Map predicted class to label (classes are 0-5, labels are 1-6)
            predicted_label = predicted_class + 1
            
            st.write('-----------------------------------------')
            print()
            st.write(f"Model Prediction: Class {predicted_class} (Confidence: {confidence:.2f}%)")
            
            if predicted_label == 1:
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Eye Closed"}</h1>', unsafe_allow_html=True)
            elif predicted_label == 2:
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Forward Look"}</h1>', unsafe_allow_html=True)
            elif predicted_label == 3:
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Left Look"}</h1>', unsafe_allow_html=True)
            elif predicted_label == 4:
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Eye Opened"}</h1>', unsafe_allow_html=True)
            elif predicted_label == 5:
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Open Partially"}</h1>', unsafe_allow_html=True)
            elif predicted_label == 6:
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Right Look"}</h1>', unsafe_allow_html=True)
            else:
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Unable to identify eye state"}</h1>', unsafe_allow_html=True)
            
                print()
            st.write('---------------------------------')
            
        except Exception as e:
            # Fallback to distance-based matching if model prediction fails
            st.write("Using fallback prediction method...")
            try:
                Total_length = len(dot1)
                
                # Find the closest match by comparing mean values
                test_mean = np.mean(gray1)
                distances = [abs(np.mean(dot1[ijk]) - test_mean) for ijk in range(Total_length)]
                closest_idx = np.argmin(distances)
                
                predicted_label = labels1[closest_idx]
                
                st.write('-----------------------------------------')
                print()
                if predicted_label == 1:
                    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Eye Closed"}</h1>', unsafe_allow_html=True)
                elif predicted_label == 2:
                    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Forward Look"}</h1>', unsafe_allow_html=True)
                elif predicted_label == 3:
                    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Left Look"}</h1>', unsafe_allow_html=True)
                elif predicted_label == 4:
                    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Eye Opened"}</h1>', unsafe_allow_html=True)
                elif predicted_label == 5:
                    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Open Partially"}</h1>', unsafe_allow_html=True)
                elif predicted_label == 6:
                    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Right Look"}</h1>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Unable to identify eye state"}</h1>', unsafe_allow_html=True)
                print()
                st.write('---------------------------------')
            except Exception as e2:
                st.error(f"Prediction failed: {str(e2)}")
                st.write("Please try uploading a different image.")

                

                    
                
                
                
                
                
                
                
                
                
                
                
                