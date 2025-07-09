'''
🚀 Introducing SnackGPT: Your AI-Powered Food Wingman 🤖🍔

I'm excited to be working on a concept I'm calling SnackGPT — a multi-agent-based intelligent application that goes way beyond traditional food delivery.

🧠 Built using LangGraph (from the LangChain family), this GenAI-powered app uses autonomous agents to:

🍱 Understand user preferences
🌦️ Adapt to real-world environmental conditions (yes, even the weather!)
📍 Recommend and route food dynamically based on location, cravings, and climate
🍕 Collaborate agentically to ensure your tastebuds get what they deserve — fast and smart.

Imagine an app that says:
"Hey, it’s raining outside. Want some hot soup or masala chai?" ☔🍲
Or:
"You’ve been eating too much cheese lately. Here’s a grilled veggie wrap with a 10-minute delivery." 🥗🕒

SnackGPT is where LLMs meet lunch — a playful blend of RAG pipelines, agent-based reasoning, and contextual intelligence all packaged into one delicious experience.
'''

import streamlit as st
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
import time
import requests
import logging
import re
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger("weather_agent")

#def main():

    # Configure Streamlit page
st.set_page_config(
        page_title="SnackGPT",
        page_icon="🌦️🍔",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
st.markdown("""
    <style>
        .main > div {
            padding-top: 1rem;
        }
        
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid;
        }
        
        .user-message {
            background-color: #f0f2f6;
            border-left-color: #4CAF50;
        }
        
        .assistant-message {
            background-color: #e8f4f8;
            border-left-color: #2196F3;
        }
        
        .therapist-message {
            background-color: #fff3e0;
            border-left-color: #FF9800;
        }
        
        .logical-message {
            background-color: #f3e5f5;
            border-left-color: #9C27B0;
        }
        
        .message-type-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .emotional-badge {
            background-color: #fff3e0;
            color: #E65100;
            border: 1px solid #FFB74D;
        }
        
        .logical-badge {
            background-color: #f3e5f5;
            color: #4A148C;
            border: 1px solid #CE93D8;
        }
        
        .weather-message {
        background-color: #e3f2fd; /* light blue */
        border-left-color: #29b6f6;
        }

        .weather-badge {
            background-color: #e1f5fe;
            color: #0277bd;
            border: 1px solid #4fc3f7;
        }    
        
        .food-message {
            background-color: #fff8e1;
            border-left-color: #ffb300;
        }
            
        .food-badge {
            background-color: #fff3e0;
            color: #e65100;
            border: 1px solid #ffb74d;
        }

        .sidebar .element-container {
            margin-bottom: 1rem;
        }
        
        .stButton > button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 0.5rem;
            border-radius: 5px;
            font-weight: bold;
        }
        
        .stButton > button:hover {
            background-color: #45a049;
        }
        
        .clear-button > button {
            background-color: #f44336;
        }
        
        .clear-button > button:hover {
            background-color: #da190b;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize the chatbot components
@st.cache_resource
def initialize_chatbot():
        """Initialize the LangGraph chatbot"""
        
        # Initialize with your API key
        llm = ChatGroq(model="llama3-8b-8192", api_key="key to be provided here")
        
        class MessageClassifier(BaseModel):
            message_type: Literal["emotional", "logical", "weather","food"] = Field(
                ...,
                description="Classify if the message requires an emotional (therapist) or logical response, or the message responds back with weather details."
            )

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            message_type: str | None

        # 1. The entry point to classify the message, if its related to Weather or something else
        def classify_message(state: State):
            last_message = state["messages"][-1]
            classifier_llm = llm.with_structured_output(MessageClassifier)

            result = classifier_llm.invoke([
                {
                    "role": "system",
                    "content": """Classify the user message as either:
                    - 'weather': if it asks about any weather updates, or any climatic conditions
                    - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
                    - 'logical': if it asks for facts, information, logical analysis, or practical solutions
                    - 'food': if the message asks for what to eat, food suggestions, or cravings based on mood/weather
                    """
                },
                {"role": "user", "content": last_message.content}
            ])
            return {"message_type": result.message_type}

        def router(state: State):
            message_type = state.get("message_type", "logical")
            print("===The message type that we receive for router", message_type)
            if message_type == "weather":
                return "weather"
            if message_type == "food":
                return "food"
            if message_type == "emotional":
                return "therapist"
            return "logical"

        # A food recommnder agent based on the Weather response
        def food_recommender_agent(state: State):
            last_message = state["messages"][-1]
            
            # Here, you assume the previous message is the weather summary
            messages = [
                SystemMessage(content="You are a food recommender that suggests dishes based on current weather conditions."),
                HumanMessage(content="Suggest a dish to enjoy in this weather:"),
                last_message
            ]
            
            reply = llm.invoke(messages)
            
            return {
                "messages": [AIMessage(content=reply.content)],
                "message_type": "food"
            }


        def extract_city_from_message(message: str) -> str:
           # Lowercase message for easier pattern matching
            msg = message.lower()

            # Match "weather in <city>", optionally followed by words like "like", "today", "now"
            match = re.search(r"weather in ([a-zA-Z\s]+?)(?:\s+like|\s+today|\s+now|[\.\?\!]|$)", msg)
            
            if match:
                city = match.group(1).strip()
                return city.title()  # Capitalize like "Bangalore"
            
            # If nothing matched, return a fallback
            return ""

        # 2. Building different agents here
        # i) Weather agent
        def weather_agent(state: State):
            print("===Entered weather agent====")
            last_message = state["messages"][-1]
            #print("the last message is ", last_message.content)
            city = extract_city_from_message(last_message.content)
            print("=========city name is==========", city)
            #breakpoint() 
            if not city:
                return {"messages": [AIMessage(content="Please mention a city to get weather details.")]}

            try:
                API_KEY = "key to be provided here"
                url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

                response = requests.get(url)
                data = response.json()

                if response.status_code == 200:
                    weather = data['weather'][0]['description'].capitalize()
                    temperature = data['main']['temp']
                    humidity = data['main']['humidity']

                    weather_summary  = (
                        f"🌦️ Weather in {city}:\n"
                        f"- Condition: {weather}\n"
                        f"- Temperature: {temperature}°C\n"
                        f"- Humidity: {humidity}%"
                    )

                    print("=====The weather summary is====", weather_summary)
                    
                    messages = [
                        SystemMessage(content="You are a weather expert who turns raw weather info into friendly advice."),
                        HumanMessage(content=f"The user asked: {last_message.content}"),
                        HumanMessage(content=f"The current weather in {city} is:\n{weather_summary}")
                    ]

                    reply = llm.invoke(messages)

                    return {
                        "messages": [AIMessage(content=reply.content)],
                        "message_type": "weather"
                    }

                else:
                    message = f"❌ Could not fetch weather for '{city}': {data.get('message', 'Unknown error')}"

            except Exception as e:
                message = f"❌ Weather API error: {str(e)}"
                st.error(f"❌ An error occurred: {str(e)}")
                st.write("Full error details:", e)

            return {
                    "messages": [AIMessage(content=reply.content)],
                    "message_type": "weather"  # ✅ Fix: Inform downstream logic
                }
        
        # ii) Therapist agent
        def therapist_agent(state: State):
            last_message = state["messages"][-1]

            messages = [
                {"role": "system",
                "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                                Show empathy, validate their feelings, and help them process their emotions.
                                Ask thoughtful questions to help them explore their feelings more deeply.
                                Avoid giving logical solutions unless explicitly asked."""
                },
                {
                    "role": "user",
                    "content": last_message.content
                }
            ]
            reply = llm.invoke(messages)
            return {"messages": [AIMessage(content=reply.content)]}

        # iii) Logical agent
        def logical_agent(state: State):
            last_message = state["messages"][-1]

            messages = [
                {"role": "system",
                "content": """You are a purely logical assistant. Focus only on facts and information.
                    Provide clear, concise answers based on logic and evidence.
                    Do not address emotions or provide emotional support.
                    Be direct and straightforward in your responses."""
                },
                {
                    "role": "user",
                    "content": last_message.content
                }
            ]
            reply = llm.invoke(messages)
            return {"messages": [AIMessage(content=reply.content)]}

        # Build the graph
        graph_builder = StateGraph(State)

        graph_builder.add_node("classifier", classify_message)
        graph_builder.add_node("weather", weather_agent)
        graph_builder.add_node("food", food_recommender_agent)
        graph_builder.add_node("therapist", therapist_agent)
        graph_builder.add_node("logical", logical_agent)

        graph_builder.add_edge(START, "classifier")
        
        graph_builder.add_conditional_edges(
            "classifier",
            router,
            {"weather": "weather","food":"food","therapist": "therapist", "logical": "logical"}
        )

        graph_builder.add_edge("weather", END)
        graph_builder.add_edge("food", END)
        graph_builder.add_edge("therapist", END)
        graph_builder.add_edge("logical", END)

        graph = graph_builder.compile()
        
        return graph

    # Initialize session state
if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
if 'graph' not in st.session_state:
        with st.spinner("🤖 Initializing chatbot..."):
            st.session_state.graph = initialize_chatbot()

    # Create layout with 30% left pane and 70% right pane
left_col, right_col = st.columns([3, 7])

    # Left pane (30%) - Controls and Info
with left_col:
        st.header("🌦️🍔SnackGPT")
        
        st.markdown("---")
        
        # Agent Type Indicator
        st.subheader("🎯 Agent Types")

        with st.expander("🍔 Food Recommeder Agent", expanded=True):
            st.markdown("""
            **Handles food related queries:**
            - Weather based food and recommendations
            - Weather based cuisines suggested
            """)
        
        with st.expander("🌦️ Weather Agent", expanded=True):
            st.markdown("""
            **Handles weather related queries:**
            - Weather based suggestions
            - Weather based conclusions
            """)
            
        with st.expander("💭 Therapist Agent", expanded=True):
            st.markdown("""
            **Handles emotional queries:**
            - Feelings and emotions
            - Personal problems
            - Therapy and support
            - Relationship issues
            """)
        
        with st.expander("🧠 Logical Agent", expanded=True):
            st.markdown("""
            **Handles logical queries:**
            - Facts and information
            - Analysis and reasoning
            - Technical questions
            - Problem-solving
            """)
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📊 Successful Deliveries")
        total_messages = len(st.session_state.chat_messages)
        user_messages = len([msg for msg in st.session_state.chat_messages if msg['role'] == 'user'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Messages", total_messages)
        with col2:
            st.metric("Your Messages", user_messages)
        
        # Message type distribution
        if st.session_state.chat_messages:
            weather_count = len([msg for msg in st.session_state.chat_messages if msg.get('type') == 'weather'])
            food_recommender_count = len([msg for msg in st.session_state.chat_messages if msg.get('type') == 'food'])
            emotional_count = len([msg for msg in st.session_state.chat_messages if msg.get('type') == 'emotional'])
            logical_count = len([msg for msg in st.session_state.chat_messages if msg.get('type') == 'logical'])
            
            if weather_count > 0 or food_recommender_count > 0 or emotional_count > 0 or logical_count > 0:
                st.markdown("**Response Types:**")
                if weather_count > 0:
                    st.write(f"🌦️ Weather: {weather_count}")
                if food_recommender_count > 0:
                    st.write(f"🍟 Food Recommender: {food_recommender_count}")    
                if emotional_count > 0:
                    st.write(f"💭 Emotional: {emotional_count}")
                if logical_count > 0:
                    st.write(f"🧠 Logical: {logical_count}")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()

    # Right pane (70%) - Chat Interface
with right_col:
        st.header("💬 Chat Interface")
        
        # Chat container
        chat_container = st.container()
        
        # Display chat messages
        with chat_container:
            for message in st.session_state.chat_messages:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Determine message type and styling
                    msg_type = message.get('type', 'logical')
                    # badge_class = "emotional-badge" if msg_type == "emotional" else "logical-badge"
                    # message_class = "therapist-message" if msg_type == "emotional" else "logical-message"
                    # agent_name = "💭 Therapist" if msg_type == "emotional" else "🧠 Logical Assistant"
                    agent_styles = {
                    "emotional": {
                        "badge_class": "emotional-badge",
                        "message_class": "therapist-message",
                        "agent_name": "💭 Therapist"
                    },
                    "logical": {
                        "badge_class": "logical-badge",
                        "message_class": "logical-message",
                        "agent_name": "🧠 Logical Assistant"
                    },
                    "weather": {
                        "badge_class": "weather-badge",  # You can define a new CSS class like `weather-badge`
                        "message_class": "weather-message",  # Or create a new one like `weather-message`
                        "agent_name": "⛅ Weather Agent"
                    },
                    "food": {
                        "badge_class": "food-badge",
                        "message_class": "food-message",
                        "agent_name": "🍟 Food Recommender Agent"
                    }
                }

                # Fallback to logical if unknown
                    style = agent_styles.get(msg_type, agent_styles["logical"])
                    st.markdown(f"""
                    <div class="chat-message {style['message_class']}">
                        <div class="message-type-badge {style['badge_class']}">
                            {style['agent_name']}
                        </div><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Input area
        st.markdown("---")
        
        # Create input form
        with st.form(key='chat_form', clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_area(
                    "Type your message:",
                    height=100,
                    placeholder="Ask me anything... I'll route you to the right agent!"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                submit_button = st.form_submit_button("Send 🍕🍥🍟")
        
        # Process user input
        if submit_button and user_input.strip():
            # Add user message to chat
            st.session_state.chat_messages.append({
                'role': 'user',
                'content': user_input
            })
            
            # Show processing message
            with st.spinner('🤖 Processing your message...'):
                try:
                    # Prepare state for the graph with proper message format
                    state = {
                        "messages": [HumanMessage(content=user_input)],
                        "message_type": None
                    }
                    
                    # Run the graph
                    result = st.session_state.graph.invoke(state)
                    
                    # Debug: Print the result structure
                    st.write("Debug - Result structure:", result)
                    
                    # Extract the response and message type
                    if result.get("messages") and len(result["messages"]) > 0:
                        # Get the last message (should be the assistant's response)
                        last_message = result["messages"][-1]
                        message_type = result.get("message_type", "logical")
                        
                        # Extract content based on message type
                        if hasattr(last_message, 'content'):
                            content = last_message.content
                        else:
                            content = str(last_message)
                        
                        # Add assistant response to chat
                        st.session_state.chat_messages.append({
                            'role': 'assistant',
                            'content': content,
                            'type': message_type
                        })
                        
                        # Success message
                        st.success(f"✅ Responded using {message_type} agent")
                        
                    else:
                        st.error("❌ No response received from the agents")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.write("Full error details:", e)
            
            # Rerun to display the new messages
            st.rerun()

    # Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🌦️🍔SnackGPT - Powered by LangGraph & Streamlit</p>
        <p><em>Your AI-Powered Food Wingman, recommeding and routing food automatically </em></p>
    </div>
    """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
