from flask import Flask, request, jsonify
from rag import RAGEngine
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyDSMgPLt9jaQM5adyTw7-zkTiJLFrRDdso")

app = Flask(__name__)
rag = RAGEngine('inara_qa.json')

# Company background context
company_context = """Inara Technologies is a software services company based in Islamabad, Pakistan. 
Founded in 2013, Inara specializes in enterprise solutions, AI automation, cloud platforms, and end-to-end tech support. 
The company has between 51 and 200 employees and serves both national and international clients."""

# Use Gemini to generate response based on retrieved FAQ + user query
def generate_gemini_response(user_query, retrieved_faqs, context):
    faqs_text = "\n".join(
        [f"Q: {faq['question']}\nA: {faq['answer']}" for faq in retrieved_faqs]
    )

    prompt = f"""
You are a helpful customer support assistant for Inara Technologies.

Company Info:
{context}

Relevant FAQ Data:
{faqs_text}

User Question:
"{user_query}"

Answer in a helpful, accurate, and conversational way:
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash") 
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("🔥 Gemini API Error:", e)
        return "Sorry, I couldn’t process that. Please try again later."

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    
    # Retrieve top 3 similar FAQ entries
    retrieved_faqs = rag.retrieve_top_k(user_input, k=3)
    
    # Generate Gemini-based response using the query + top FAQs + context
    ai_response = generate_gemini_response(user_input, retrieved_faqs, company_context)
    
    return jsonify({'response': ai_response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
