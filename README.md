# Satellite-Imaging-Segmentation-Using-Deep-Learning

**Overview**

This is a deep learning-based project designed to perform segmentation on satellite imagery. Using the U-Net architecture, the project accurately identifies and classifies various land cover features within satellite images. The model is trained to segment complex satellite images and is packaged with an intuitive interface built using Gradio and Hugging Face, allowing for seamless interaction and visualization of the segmentation results. Additionally, the project is Colab-compatible, making it easy to run and experiment with.

**Features**

**1**- Deep Learning Architecture: Utilizes the U-Net model, a powerful convolutional network architecture for fast and precise image segmentation.

**2**- Interactive Interface: The project leverages Gradio and Hugging Face for a user-friendly UI/UX, enabling users to upload images and view segmented outputs in real time.

**3**- Colab Compatibility: The project can be easily run on Google Colab, providing a cloud-based environment for experimentation and development.

**4**- High Accuracy: Trained on satellite imagery datasets, the model offers precise segmentation results for various land cover types.

**Installation**

To run the project, follow the steps below:

**1**. Clone the Repository
   
>> git clone https://github.com/yourusername/skypixel.git
>> cd skypixel

**2**. Create and Activate a Virtual Environment
   
It's recommended to use a virtual environment to manage dependencies.
>> python -m venv venv
>> source venv/bin/activate  # On Windows use `venv\Scripts\activate`

**3**. Install Dependencies

Install the necessary Python packages listed in the requirements.txt file.

>> pip install -r requirements.txt

**4**. Download the Pre-trained Model

Place the provided unet_model.h5 file in the project directory.

**5**. Run the Application

You can run the application using the following command:

>> gradio app App_satellite_segmentation_prediction.ipynb

This will launch a web interface where you can upload satellite images and get segmented outputs.

**6**. Run on Google Colab

If you prefer to run the project on Google Colab, you can easily upload the notebooks and model files to Colab, install the required dependencies, and execute the cells. This allows for a convenient cloud-based experience.

**Usage**

Open the application in your browser.

Upload a satellite image.

View the segmentation result displayed on the interface.

**Contributing**

If you would like to contribute to this project, feel free to open issues or submit pull requests. We welcome contributions that can help improve the accuracy, efficiency, or usability of SkyPixel.

**Authors**

Laraib Khalid 
