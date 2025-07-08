# app.py
import streamlit as st
import pandas as pd
import os

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_groq import ChatGroq
from langchain.tools import StructuredTool
from pydantic import BaseModel

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
load_dotenv()


# ---- SETUP ----
# Create synthetic Excel if it doesn't exist
# EXCEL_PATH = "user_data.xlsx"
# if not os.path.exists(EXCEL_PATH):
data = {
        "Username": ["Krish", "Anaya", "Rahul", "Meera", "Arjun", "Sara", "Rishi", "Tanya", "Dev", "Isha","Vinay"],
        "Email": [
            "krish@email.com", "anaya@email.com", "rahul@email.com", "meera@email.com", "arjun@email.com",
            "sara@email.com", "rishi@email.com", "tanya@email.com", "dev@email.com", "isha@email.com","vishvedula@gmail.com"
        ],
        "Preferences": [
            "Sports, Music", "Reading, Painting", "Gaming, Travel", "Yoga, Cooking", "Running, Chess",
            "Painting, Music", "Trekking, Movies", "Reading, Dance", "Cricket, Cooking", "Travel, Sports",
            "Football, Painting"
        ],
        "Weather": ["Sunny", "Rainy", "Windy", "Cloudy", "Snowy", "Rainy", "Sunny", "Windy", "Cloudy", "Sunny", "Rainy"],
        "Food": ["Pizza", "Biryani", "Pasta", "Sushi", "Pizza", "Burgers", "Salad", "Sandwich", "Biryani", "Pasta","Biryani"]
    }
pd.DataFrame(data).to_csv("user_data.csv", index=False)
user_df = pd.read_csv("user_data.csv")

# ---- TOOL LOGIC ----
def search_user_data(query: str) -> dict:
    query_lower = query.lower()

    matched_user = None
    for name in user_df["Username"]:
        if name.lower() in query_lower:
            matched_user = name
            break

    if matched_user:
        user_row = user_df[user_df["Username"] == matched_user]
        email = user_row["Email"].values[0]

        if "food" in query_lower:
            food = user_row["Food"].values[0]
            summary = f"{matched_user} likes to eat {food}."
            return {"summary": summary, "email": email}

        elif "weather" in query_lower:
            weather = user_row["Weather"].values[0]
            summary = f"{matched_user}'s weather preference is {weather}."
            return {"summary": summary, "email": email}

        elif "email" in query_lower:
            summary = f"{matched_user}'s email is {email}."
            return {"summary": summary, "email": email}

        elif "preferences" in query_lower:
            prefs = user_row["Preferences"].values[0]
            summary = f"{matched_user}'s preferences are: {prefs}."
            return {"summary": summary, "email": email}

        else:
            summary = f"⚠️ Found user '{matched_user}', but couldn't identify which attribute you're asking for."
            return {"summary": summary, "email": email}

    # Combo: food and weather
    elif any(food in query_lower for food in ["pizza", "biryani", "pasta", "burgers", "salad", "sushi", "sandwich"]) and \
         any(weather in query_lower for weather in ["sunny", "rainy", "cloudy", "windy", "snowy"]):
        
        food_match = next((food for food in ["pizza", "biryani", "pasta", "burgers", "salad", "sushi", "sandwich"] if food in query_lower), None)
        weather_match = next((weather for weather in ["sunny", "rainy", "cloudy", "windy", "snowy"] if weather in query_lower), None)

        result = user_df[
            (user_df["Food"].str.lower().str.contains(food_match, na=False)) &
            (user_df["Weather"].str.lower().str.contains(weather_match, na=False))
        ]

        if result.empty:
            return {"summary": "⚠️ No user found with that food and weather preference.", "email": None}
        
        usernames = ", ".join(result["Username"].tolist())
        emails = ", ".join(result["Email"].tolist())
        summary = f"Users who like {food_match} and prefer {weather_match} weather: {usernames}."
        return {"summary": summary, "email": emails}

    # Individual food
    elif "pizza" in query_lower:
        result = user_df[user_df["Food"].str.lower().str.contains("pizza", na=False)]
    elif "biryani" in query_lower:
        result = user_df[user_df["Food"].str.lower().str.contains("biryani", na=False)]
    
    # Individual weather
    elif "weather" in query_lower:
        result = user_df[["Username", "Weather"]]
    
    # Individual email
    elif "email" in query_lower:
        result = user_df[["Username", "Email"]]
    
    # Preferences (like cricket)
    elif "cricket" in query_lower:
        result = user_df[user_df["Preferences"].str.lower().str.contains("cricket", na=False)]
    
    else:
        return {"summary": "⚠️ I couldn't find relevant data for your query.", "email": None}

    if result.empty:
        return {"summary": "⚠️ No matching user found.", "email": None}

    # Return full table as string, and None for email since it's multi-user
    return {"summary": result.to_string(index=False), "email": None}


class ExcelQueryInput(BaseModel):
    query: str

excel_user_lookup = StructuredTool.from_function(
    func=search_user_data,
    name="excel_user_lookup",
    description="""
    This tool looks up user information from an Excel file. 
    Use natural language queries like:
    - 'Who likes pizza?'
    - 'Who likes cricket?'
    - 'What is one's weather preference?'
    - 'Show me the emails of all users'
    """,
    args_schema=ExcelQueryInput
)

# ---- LLM SETUP ----
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama3-8b-8192", api_key="key")
llm_with_tool = llm.bind_tools([excel_user_lookup])

# ---- LANGGRAPH ----
class State(TypedDict):
    messages: Annotated[list, add_messages]

def tool_calling_llm(state: State):
    result = llm_with_tool.invoke(state["messages"])
    return {"messages": [result]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([excel_user_lookup]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Excel RAG Bot", page_icon="📊")
st.title("📊 Excel RAG Chatbot")

st.markdown("Ask questions like:")
st.code("Who likes pizza?\nWhat is Arjun's weather preference?\nShow emails of all users")

# ✅ Streamlit App
st.title("📧 Ask User Preferences + Send Email")

# ✅ Email Sender
def send_email(to_email, subject, body, sender_email, sender_password):
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP("smtp.mail.yahoo.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        return f"✅ Email sent to {to_email}"
    except Exception as e:
        return f"❌ Failed to send email: {e}"

query = st.text_input("Ask a question about food/weather/email:", placeholder="e.g., What does Dev eat when it's rainy?")
#send_clicked = st.button("Send Email")

if query:
    with st.spinner("Thinking..."):
        response = graph.invoke({"messages": query})
        answer = response["messages"][-1].content
        st.success("💬 Bot Response:")
        st.markdown(answer)

        # Try to fetch email info from tool directly
        tool_output = search_user_data(query)
        email = tool_output.get("email")
        summary = tool_output.get("summary")

        if email:
            st.info(f"📨 Matched Email: `{email}`")

            # Show input fields BEFORE button is pressed
            sender_email = "user@yahoo.co.in"
            sender_password = "pwd"

            if st.button("Send Email"):
                if sender_email and sender_password:
                    result = send_email(
                        to_email=email,
                        subject="User Preference Info",
                        body=summary,
                        sender_email=sender_email,
                        sender_password=sender_password
                    )
                    st.success(result)
                else:
                    st.warning("⚠️ Please enter sender email & password before sending.")
        else:
            st.warning("No email matched from the query.")



