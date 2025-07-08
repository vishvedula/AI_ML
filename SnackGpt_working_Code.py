# 📦 Imports
import pandas as pd
from typing import TypedDict, Annotated
from typing_extensions import TypedDict
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
import os
from langchain.tools import StructuredTool
from pydantic import BaseModel
from IPython.display import Image,display


# 📁 Create Synthetic Excel File
data = {
    "Username": ["Krish", "Anaya", "Rahul", "Meera", "Arjun", "Sara", "Rishi", "Tanya", "Dev", "Isha","Vinay"],
    "Email": [
        "krish@email.com", "anaya@email.com", "rahul@email.com", "meera@email.com", "arjun@email.com",
        "sara@email.com", "rishi@email.com", "tanya@email.com", "dev@email.com", "isha@email.com","vishvedula@gmail.com"
    ],
    "Preferences": [
        "Sports, Music", "Reading, Painting", "Gaming, Travel", "Yoga, Cooking", "Running, Chess",
        "Painting, Music", "Trekking, Movies", "Reading, Dance", "Cricket, Cooking", "Travel, Sports","Travel ,Painting"
    ],
    "Weather": ["Sunny", "Rainy", "Windy", "Cloudy", "Snowy", "Rainy", "Sunny", "Windy", "Cloudy", "Sunny", "Rainy"],
    "Food": ["Pizza", "Biryani", "Pasta", "Sushi", "Pizza", "Burgers", "Salad", "Sandwich", "Biryani", "Pasta", "Samosa"]
}
df = pd.DataFrame(data)
df.to_excel("user_data.xlsx", index=False)
print("✅ 'user_data.xlsx' created.")

# 📥 Load Excel
user_df = pd.read_excel("user_data.xlsx")

# 🤖 Define Tool Logic
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

'''@tool
def excel_user_lookup(query: str) -> str:
    """Search user info from Excel file based on a natural language query"""
    return search_user_data(query)'''



# Define schema for the tool input
class ExcelQueryInput(BaseModel):
    query: str

# Use StructuredTool instead of @tool
excel_user_lookup = StructuredTool.from_function(
    func=search_user_data,
    name="excel_user_lookup",
    description="""
    This tool looks up user information from an Excel file. 
    It supports natural language queries like:
    - 'Who likes pizza?'
    - 'Who likes cricket?'
    - 'What is ones weather preference?'
    - 'Show me the emails of all users'
    - 'Show ones weather and food preference'

    It returns matching rows from the Excel file.
    """,
    args_schema=ExcelQueryInput
)

# Define tools and bind to LLM
tools = [excel_user_lookup]
llm_with_tool = llm.bind_tools(tools)
# 🧠 Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 🔑 Set your Groq key (or use environment variable)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "<key>")
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

# 🛠️ Bind the tool to LLM
tools = [excel_user_lookup]
llm_with_tool = llm.bind_tools(tools)

# 🔧 Define LangGraph nodes
def tool_calling_llm(state: State):
    result = llm_with_tool.invoke(state["messages"])
    #print("🛠️ Tool call result:\n", result)  # Debug output
    return {"messages": [result]}

# 🕸️ Build the LangGraph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")  # Loop if tools are called

graph = builder.compile()

# Verify the image file path and ensure the graph visualization is displayed correctly
try:
    display(Image(graph.get_graph().draw_mermaid_png()))  # Check if the graph visualization is generated properly
    #print(graph.get_graph())  # Print the Mermaid graph representation
except Exception as e:
    print(f"An error occurred: {e}")

# ✅ Test Sample Queries
print("\n--- Test 1: Who likes pizza? ---")
response = graph.invoke({"messages": "Who likes pizza, give me any one username?"})
print(response["messages"][-1].content)

print("\n--- Test 2: What is Arjun's weather preference? ---")
response = graph.invoke({"messages": "What is Arjun's weather preference?"})
print(response["messages"][-1].content)

print("\n--- Test 3: What does Vinay like when its rainy? ---")
response = graph.invoke({"messages": "What does Vinay like when its rainy"})
print(response["messages"][-1].content)

# print("\n--- Test 3: Show me users who like cricket ---")
# response = graph.invoke({"messages": "Who likes cricket?"})
# print(response["messages"][-1].content)

# print("\n--- Test 3: What are the emails of users? ---")
# response = graph.invoke({"messages": "Give me email addresses"})
# print(response["messages"][-1].content)

# print("\n--- Test 4: What weather does each user prefer? ---")
# response = graph.invoke({"messages": "What is the most preferred weather of users?"})
# print(response["messages"][-1].content)


#===========
# Response
'''
--- Test 1: Who likes pizza? ---
Arjun

--- Test 2: What is Arjun's weather preference? ---
Arjun's weather preference is Snowy.

--- Test 3: What does Vinay like when its rainy? ---
Vinay likes Samosa when it is rainy.

'''
