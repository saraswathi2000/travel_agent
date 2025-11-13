import streamlit as st
import os
import json
from typing import Dict, Any
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from tools import (
    load_sheet_data, 
    find_flights, 
    find_hotels,
    save_message_to_sheet,
    load_chat_history_from_sheet,
    clear_sheet_history,
    get_all_sessions
)

# Configuration
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SHEET_NAME = "AI_Agent_data"

# Page configuration
st.set_page_config(
    page_title="Travel Planner Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #202123;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ececf1;
    }
    
    .stButton button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
        background-color: transparent;
        color: white;
        padding: 10px;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: rgba(255,255,255,0.1);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stChatInput {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.3
    )

llm = get_llm()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False

if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = ["default"]




def check_input_guardrails(user_input: str) -> tuple[bool, str]:
    """
    Check if user input violates guardrails
    Returns: (is_valid, error_message)
    """
    
    # Check 1: Input length
    if len(user_input.strip()) < 3:
        return False, " Please provide more details about your travel plans."
    
    if len(user_input) > 1000:
        return False, " Your message is too long. Please keep it under 1000 characters."
    
    # Check 2: Inappropriate content (basic filters)
    inappropriate_keywords = [
        'hack', 'exploit', 'illegal', 'drugs', 'weapons',
        'violence', 'terrorism', 'steal', 'fraud'
    ]
    
    user_input_lower = user_input.lower()
    for keyword in inappropriate_keywords:
        if keyword in user_input_lower:
            return False, " I'm a travel planning assistant. Please ask travel-related questions only."
    
    # Check 3: Non-travel queries (basic detection)
    non_travel_keywords = [
        'recipe', 'code', 'program', 'python', 'javascript', 
        'medicine', 'disease', 'legal advice', 'financial advice',
        'homework', 'essay', 'write me'
    ]
    
    # Only flag if it's clearly not travel-related
    travel_keywords = [
        'trip', 'travel', 'flight', 'hotel', 'vacation', 'visit',
        'tour', 'destination', 'booking', 'budget', 'itinerary',
        'airport', 'plan', 'go to', 'want to', 'going to'
    ]
    
    has_travel_context = any(keyword in user_input_lower for keyword in travel_keywords)
    has_non_travel = any(keyword in user_input_lower for keyword in non_travel_keywords)
    
    if has_non_travel and not has_travel_context:
        return False, " I'm specialized in travel planning. Please ask me about trips, destinations, flights, hotels, or travel activities."
    
    # Check 4: Spam detection (repeated characters/words)
    if re.search(r'(.)\1{10,}', user_input):  # 10+ repeated characters
        return False, " Please provide a valid travel query."
    
    return True, ""


def check_output_guardrails(ai_response: str, user_input: str) -> tuple[bool, str]:
    """
    Check if AI response is appropriate and travel-related
    Returns: (is_valid, error_message)
    """
    
    # Check 1: Response length
    if len(ai_response.strip()) < 50:
        return False, " Response too short. Let me provide more details."
    
    # Check 2: Check if response is actually about travel
    travel_indicators = [
        'flight', 'hotel', 'destination', 'trip', 'travel',
        'itinerary', 'budget', 'vacation', 'visit', 'tour',
        'day', 'activities', 'airport', 'accommodation'
    ]
    
    response_lower = ai_response.lower()
    has_travel_content = any(indicator in response_lower for indicator in travel_indicators)
    
    if not has_travel_content and len(st.session_state.messages) > 0:
        # Allow initial greetings, but subsequent messages should be travel-related
        return False, " Let me refocus on your travel plans. Could you tell me more about your trip?"
    
    # Check 3: Ensure no harmful/inappropriate content in response
    harmful_patterns = [
        'illegal', 'dangerous', 'unsafe', 'scam', 'fraud'
    ]
    
    if any(pattern in response_lower for pattern in harmful_patterns):
        return False, " I apologize, but I can only provide safe and legal travel advice."
    
    return True, ""


def validate_budget(budget_str: str) -> tuple[bool, str]:
    """Validate budget input"""
    try:
        budget = float(budget_str)
        if budget < 100:
            return False, " Budget seems too low for a realistic trip. Please provide a budget of at least $100."
        if budget > 1000000:
            return False, " Budget seems unrealistically high. Please provide a more reasonable budget."
        return True, ""
    except:
        return True, ""  # Let LLM handle parsing


def sanitize_input(user_input: str) -> str:
    """Sanitize user input"""
    # Remove excessive whitespace
    user_input = ' '.join(user_input.split())
    
    # Remove any potential code injection attempts
    user_input = user_input.replace('<script>', '').replace('</script>', '')
    user_input = user_input.replace('<?php', '').replace('?>', '')
    
    return user_input.strip()


def format_chat_history() -> str:
    """Format chat history as a readable string"""
    messages = st.session_state.chat_history.messages
    if not messages:
        return "No previous conversation."

    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")

    return "\n".join(formatted)


def restore_chat_history(session_id: str = "default"):
    """Restore chat history from Google Sheets into memory"""
    messages = load_chat_history_from_sheet(SHEET_NAME, session_id)
    
    st.session_state.chat_history.clear()
    st.session_state.messages = []
    
    for msg in messages:
        role = msg.get("Role", "")
        content = msg.get("Content", "")
        
        if role == "user":
            st.session_state.chat_history.add_user_message(content)
            st.session_state.messages.append({"role": "user", "content": content})
        elif role == "assistant":
            st.session_state.chat_history.add_ai_message(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
    
    return len(messages)


def clear_history(session_id: str = "default"):
    """Clear chat history from both memory and Google Sheets"""
    st.session_state.chat_history.clear()
    st.session_state.messages = []
    clear_sheet_history(SHEET_NAME, session_id)


def load_all_sessions():
    """Load all available sessions"""
    sessions = get_all_sessions(SHEET_NAME)
    if not sessions:
        sessions = ["default"]
    st.session_state.all_sessions = sessions
    return sessions


def switch_session(session_id: str):
    """Switch to a different session"""
    st.session_state.session_id = session_id
    st.session_state.history_loaded = False
    restore_chat_history(session_id)


def create_new_session():
    """Create a new session"""
    import time
    new_session_id = f"trip_{int(time.time())}"
    st.session_state.session_id = new_session_id
    st.session_state.chat_history.clear()
    st.session_state.messages = []
    st.session_state.history_loaded = True
    if new_session_id not in st.session_state.all_sessions:
        st.session_state.all_sessions.append(new_session_id)




# System guardrail prompt
system_guardrail = """
CRITICAL SYSTEM INSTRUCTIONS - YOU MUST FOLLOW THESE:

1. YOU ARE ONLY A TRAVEL PLANNING ASSISTANT
   - Only answer questions about travel, trips, destinations, flights, hotels, activities
   - Politely decline any non-travel questions
   - Do not provide information on: coding, medicine, legal advice, financial advice, homework, etc.

2. SAFETY AND ETHICS
   - Never suggest illegal activities
   - Always prioritize traveler safety
   - Do not provide information about dangerous locations without warnings
   - Do not help with fraudulent activities

3. STAY IN SCOPE
   - Focus on: destinations, flights, hotels, itineraries, budgets, activities
   - Redirect off-topic questions back to travel planning

4. BE HELPFUL BUT CAUTIOUS
   - Provide realistic travel advice
   - Acknowledge limitations (e.g., "I can't book flights, but I can help you find options")
   - Don't make up information about flights, hotels, or prices not in the data

If user asks non-travel questions, respond with:
"I'm specialized in travel planning. I can help you with trip planning, destinations, flights, hotels, and activities. How can I assist with your travel plans?"
"""

extract_prompt = ChatPromptTemplate.from_template(system_guardrail + """

You are an assistant that extracts structured travel details from a user's request.

Conversation so far:
{chat_history}

User Request:
{user_text}

IMPORTANT INSTRUCTIONS:
1. If this is a NEW trip request, extract all available details
2. If this is a FOLLOW-UP question about an existing trip, preserve previous trip details
3. If user asks NON-TRAVEL questions, set query_type to "off_topic"

Return a valid JSON with keys:
origin_city, destination_city, start_date, end_date, trip_length_days, budget_usd, interests, query_type, needs_clarification, clarification_question, missing_fields

query_type options: "new_trip", "providing_details", "hotel_query", "flight_query", "activity_query", "budget_query", "general_query", "modification", "off_topic"

If query_type is "off_topic", leave all other fields null.
""")

extract_chain = RunnableSequence(extract_prompt | llm | StrOutputParser())

summary_prompt = ChatPromptTemplate.from_template(system_guardrail + """

You are a helpful travel planner. Use the chat history and structured data below to provide a contextual response.

Chat History:
{chat_history}

Structured Data:
{final_state}

User's Current Request:
{user_text}

RESPONSE INSTRUCTIONS:
1. **Analyze what the user is specifically asking for**
2. **If it's a new trip**: Provide complete itinerary (flights, hotels, activities, budget)
3. **If it's a follow-up about specific aspect**: Focus ONLY on that aspect
4. **If query_type is "off_topic"**: Politely redirect to travel planning

Stay focused, relevant, and helpful. Only discuss travel-related topics.
""")

summary_chain = RunnableSequence(summary_prompt | llm | StrOutputParser())




def simulate_tool_calls(structured_json_str: str) -> Dict[str, Any]:
    import json
    """Simulate tool calls to fetch flight and hotel data"""
    st.write('1')
    st.write(type(structured_json_str))
    st.write(structured_json_str)
    st.write('2')
    if isinstance(structured_json_str, str):
        try:
            st.write('structured_Data1')
            st.write(structured_json_str)
            structured_data = json.loads(structured_json_str)
            st.write('structured_data')
            st.write(structured_data)
        except json.JSONDecodeError:
            structured_data = {}
    elif isinstance(structured_json_str, dict):
        structured_data = structured_json_str
    else:
        structured_data = {}
    st.write(3)
    st.write(structured_data)
    # try:
    #     st.write('here')
    #     structured_data = json.loads(structured_json_str)
    #     # structured_data = structured_json_str
    #     st.write('nowhere')
    #     st.write(structured_json)
    #     st.write('structured json')
    # except json.JSONDecodeError:
    #     structured_data = {}

    if structured_data.get("query_type") == "off_topic":
        st.write('offtopic')
        return {"final_state": json.dumps(structured_data, indent=2)}
    st.write(3)
    st.write(structured_data)
    st.write(4)
    origin = structured_data.get("origin_city") or ""
    st.write('orgin')
    st.write(orgin)
    destination = structured_data.get("destination_city") or ""
    st.write('destination')
    st.write(destination)
    start_date = structured_data.get("start_date")
    st.write(start_date)
    
    budget = structured_data.get("budget_usd")
    st.write(budget)
    nights = structured_data.get("trip_length_days") or 3
    st.write(nights)
    budget_per_night = None

    if budget:
        try:
            budget = float(budget)
            # Budget validation
            if budget < 100 or budget > 1000000:
                structured_data["budget_warning"] = "Budget seems unusual. Please verify."
            else:
                budget_per_night = (budget * 0.4) / nights
        except Exception:
            pass

    try:
        flights_df = load_sheet_data(SHEET_NAME, "flights_data")
        st.write('flights_df',flights_df)
        hotels_df = load_sheet_data(SHEET_NAME, "hotels_data")
        st.write(hotels_df)

        flights = find_flights(flights_df, origin, destination, prefer_date=start_date, budget_usd=budget)
        st.write('flights')
        st.write(flights)
        hotels = find_hotels(hotels_df, destination, budget_per_night=budget_per_night)
        st.write('hotels')
        st.write(hotels)
    except Exception as e:
        flights = []
        hotels = []

    structured_data["tool_results"] = {
        "flight_options": flights,
        "hotel_options": hotels
    }
    st.write('structured_data')
    st.write(structured_data)
    return {"final_state": json.dumps(structured_data, indent=2)}




def run_agent(user_text: str, session_id: str = "default") -> str:
    """Main agent workflow with guardrails"""
    
    # Sanitize input
    user_text = sanitize_input(user_text)
    
    # Check input guardrails
    is_valid, error_msg = check_input_guardrails(user_text)
    if not is_valid:
        return error_msg
    
    # Rate limiting (max 50 messages per session)
    if len(st.session_state.messages) >= 100:
        return "You've reached the maximum number of messages for this session. Please start a new trip."
    
    st.session_state.chat_history.add_user_message(user_text)
    save_message_to_sheet(SHEET_NAME, "user", user_text, session_id)

    formatted_history = format_chat_history()

    try:
        with st.spinner(" Analyzing your request..."):
            structured_data = extract_chain.invoke({
                "user_text": user_text,
                "chat_history": formatted_history
            })

        with st.spinner("Finding best options..."):
            tool_output = simulate_tool_calls(structured_data)

        with st.spinner(" Preparing your plan..."):
            final_output = summary_chain.invoke({
                "final_state": tool_output["final_state"],
                "chat_history": formatted_history,
                "user_text": user_text
            })
        
        # Check output guardrails
        is_valid, error_msg = check_output_guardrails(final_output, user_text)
        if not is_valid:
            final_output = "I'm here to help with your travel planning. What destination are you interested in?"

        st.session_state.chat_history.add_ai_message(final_output)
        save_message_to_sheet(SHEET_NAME, "assistant", final_output, session_id)

        return final_output
        
    except Exception as e:
        error_response = " I encountered an error processing your request. Please try rephrasing your question."
        st.session_state.chat_history.add_ai_message(error_response)
        save_message_to_sheet(SHEET_NAME, "assistant", error_response, session_id)
        return error_response




with st.sidebar:
    if st.button(" Clear Current Trip", use_container_width=True):
        clear_history(st.session_state.session_id)
        st.success(" Cleared!")
        st.rerun()




st.title(" Travel Planner")
# st.caption("Plan your perfect trip with AI assistance")

# Load history on first run
if not st.session_state.history_loaded:
    with st.spinner("Loading conversation..."):
        count = restore_chat_history(st.session_state.session_id)
    st.session_state.history_loaded = True

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with guardrails
if prompt := st.chat_input("Where would you like to go? "):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response with guardrails
    try:
        response = run_agent(prompt, st.session_state.session_id)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    except Exception as e:
        error_msg = " Something went wrong. Please try again."
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

