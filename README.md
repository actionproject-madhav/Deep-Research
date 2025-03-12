# **Deep-Research** 🧠🔍

Tired of AI models giving shallow, surface-level answers? **Deep-Research** is an advanced research agent designed to **think deeper and longer** by systematically breaking down complex topics into **subquestions** and analyzing them thoroughly.

### **Key Features**
- **Multi-Step Thinking** – Generates in-depth insights by carefully structuring its thought process.
-  **Google API Integration** – Retrieves real-time, high-quality information from the web.
-  **Pinecone Database** – Stores and retrieves research context efficiently.
-  **OpenAI API** – Leverages powerful LLMs for reasoning and synthesis.

With **Deep-Research**, you get **meaningful, well-reasoned, and structured research**—not just quick, shallow AI-generated responses.

---
<img width="869" alt="Screenshot 2025-03-08 at 8 25 19 PM" src="https://github.com/user-attachments/assets/d4c05842-fc17-487a-89e0-46728888c06a" />
<img width="779" alt="Screenshot 2025-03-08 at 8 25 39 PM" src="https://github.com/user-attachments/assets/04a76c1a-3e11-45fb-8290-5a2ac298c260" />




## **Getting Started** 🚀

### **Prerequisites**  
To use **Deep-Research**, you’ll need the following tools and accounts:

1. **Python 3.x** – Ensure you have Python installed. You can download it from [here](https://www.python.org/downloads/).
2. **Git** – You'll need Git to clone the repository. Install it from [here](https://git-scm.com/downloads).
3. **Google API Key** – Create a Google Custom Search Engine and get an API key. [Learn how to get the API key here](https://developers.google.com/custom-search/v1/overview).
4. **Pinecone API Key** – Sign up for Pinecone and get an API key. [Get your API key here](https://www.pinecone.io/start/).
5. **OpenAI API Key** – Get your OpenAI API key by signing up at [OpenAI](https://beta.openai.com/signup/).
6. **Voyage API Key** – Obtain the API key from [Voyage](https://www.voyage.com).

---

### **Installation**

1. **Clone the Repository**  
   First, clone the repository to your local machine:
   ```bash
   git clone https://github.com/actionproject-madhav/Deep-Research.git
   cd Deep-Research
   ```

2. **Make a `.env` file**  
   Create a `.env` file in the root directory and add the following API keys and environment variables:
   ```bash
   PINECONE_API_KEY="your key here"
   PINECONE_ENV="add your region here"
   OPENAI_API_KEY="your key here"
   GOOGLE_API_KEY="your key here"
   GOOGLE_SEARCH_ENGINE_ID="your Google search engine id here"
   VOYAGE_API_KEY="your voyage api_key here"
   ```

3. **Install Dependencies**  
   Install the required dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

---

### **Usage**

1. **Run the Code**  
   After setting up your environment and keys, you can run the **Deep-Research** agent:
   ```bash
   python3 test.py
   ```

2. **Enter Your Research Query**  
   Once the script is running, you'll be prompted to the localhost link for chatbot. The agent will then break down the topic, retrieve relevant data, and give in-depth, structured responses.


### **Customizing Your Search**  
You can adjust the search parameters in the script. For example, you can modify:
- **Search Depth** – The number of results to retrieve from Google.
- **Processing Power** – Choose different models in OpenAI for stronger or quicker responses.


---

### **Troubleshooting**

- **Error: API Key Missing/Incorrect**  
  Ensure that your `.env` file has the correct keys and that they are properly formatted.
  
- **Error: ModuleNotFoundError**  
  Ensure that you have activated your virtual environment and installed the dependencies using `pip`.

---

### **Contributing**

Feel free to fork the repository and submit issues or pull requests. To contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and commit them.
4. Push your branch to your forked repository.
5. Open a pull request to the main repository.

---

### **License**

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---



