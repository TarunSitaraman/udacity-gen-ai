from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    client = OpenAI(api_key=openai_key)
    
    system_prompt = (
        "You are an expert NASA mission assistant. Use the following context to answer the user's question. "
        "If the answer is not in the context, say that you don't know based on the provided documents. "
        "Do not use outside knowledge.\n\n"
        f"Context:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    messages.extend(conversation_history)
    
    # Add the current user message
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content