How to Set Up and Run the Project (Terminal Steps)

1. Create & Activate Virtual Environment
bash
Copy
Edit
# For Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# For Windows (Command Prompt):
python -m venv venv
venv\Scripts\activate

2. Install Required Packages
pip install -r requirements.txt

3. Add Your API Key
Open your views.py file and configure your API key:
# Example:
# genai.configure(api_key=os.getenv("your_key"))  # Add your API key here
Replace "your_key" with your actual API key, or set it in environment variables for better security.

4. Run the Django Server
Navigate to the folder where manage.py is located:
cd your_project_folder
python manage.py runserver
This will start the development server at: http://127.0.0.1:8000



 
 
