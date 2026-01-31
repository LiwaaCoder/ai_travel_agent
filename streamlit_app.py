import asyncio
import os
import re

import streamlit as st
from dotenv import load_dotenv

from src.agents.graph import build_travel_agent

load_dotenv()
st.set_page_config(
    page_title="AI Travel Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main Background adjustments */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Metrics Styles */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    /* Duolingo-style Timeline */
    .timeline-container {
        position: relative;
        padding-left: 30px;
        margin-top: 20px;
    }
    
    /* The vertical line */
    .timeline-container::before {
        content: '';
        position: absolute;
        left: 14px; /* Center of the 30px badge */
        top: 10px;
        bottom: 0;
        width: 4px;
        background-color: #e5e5e5;
        border-radius: 2px;
    }
    
    /* Day Block Container */
    .day-block {
        position: relative;
        margin-bottom: 25px;
    }
    
    /* Number Badge */
    .day-badge {
        position: absolute;
        left: -32px;
        top: 0;
        width: 32px;
        height: 32px;
        background-color: #58cc02; /* Green */
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        font-weight: bold;
        border: 4px solid #ffffff; /* White halo to separate from line */
        z-index: 10;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* The Interactive Card (HTML details) */
    details.day-card {
        background-color: #ffffff;
        border-radius: 16px;
        border: 2px solid #e5e5e5;
        box-shadow: 0 4px 0 #e5e5e5; /* 3D effect */
        overflow: hidden;
        transition: all 0.2s ease;
    }
    
    details.day-card:hover {
        border-color: #dce0e3;
        top: -1px;
    }
    
    details.day-card[open] {
        border-color: #58cc02;
        box-shadow: 0 4px 0 #58cc02;
    }
    
    summary.day-header {
        padding: 15px 20px;
        font-weight: 800;
        font-size: 1.1em;
        color: #4b4b4b;
        cursor: pointer;
        list-style: none;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    summary.day-header::-webkit-details-marker {
        display: none;
    }
    
    summary.day-header:hover {
        background-color: #f7f9fa;
    }
    
    .day-content {
        padding: 0 20px 20px 20px;
        border-top: 1px solid #f0f0f0;
        animation: slideDown 0.3s ease-out;
    }
    
    /* Activity Items */
    .activity-row {
        display: flex;
        margin-top: 15px;
        align-items: flex-start;
    }
    
    .activity-icon {
        font-size: 1.2em;
        margin-right: 15px;
        min-width: 30px;
        text-align: center;
    }
    
    .activity-details strong {
        color: #1cb0f6; /* Blue highlight */
        display: block;
        margin-bottom: 2px;
        font-size: 0.95em;
    }
    
    .activity-details span {
        color: #3d3d3d;
        font-size: 1.05em;
        line-height: 1.6;
    }

    /* Global font improvements */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        font-size: 16px !important;
        line-height: 1.7 !important;
    }
    
    .stExpander {
        font-size: 16px !important;
    }
    
    .day-title {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #1a1a1a !important;
    }
    
    .time-slot {
        color: #1cb0f6 !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    
    .activity-text {
        font-size: 15px !important;
        color: #333 !important;
        line-height: 1.6 !important;
    }

    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


def parse_itinerary(itinerary_text: str):
    """Parse itinerary markdown into structured days and bullets.

    Expected format (current LLM output):
    ## Day X: Title
    - **Morning ...** detail
    - **Lunch ...** detail
    - **Afternoon ...** detail
    - **Evening ...** detail
    """
    days = []
    blocks = re.split(r"##\s*Day\s+\d+", itinerary_text)
    headers = re.findall(r"##\s*Day\s+\d+[^\n]*", itinerary_text)
    for header, body in zip(headers, blocks[1:]):
        title = header.replace("##", "").strip()
        activities = []
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("-"):
                # Extract bold label if present
                m = re.match(r"-\s*\*\*(.*?)\*\*\s*[:\-]?\s*(.*)", line)
                if m:
                    slot, desc = m.groups()
                    # Only add if there's actual content
                    if slot.strip() or desc.strip():
                        activities.append((slot.strip(), desc.strip()))
                else:
                    content = line.lstrip("- ").strip()
                    # Only add non-empty content
                    if content:
                        activities.append(("", content))
        # Only add days that have activities
        if activities:
            days.append({"title": title, "activities": activities})
    return days

# Initialize agent once (cached for speed)
@st.cache_resource
def get_agent():
    return build_travel_agent()

agent = get_agent()

# Header
st.title("✈️ AI Travel Agent")
st.markdown("""
*RAG-powered travel planning using LangChain + LangGraph*

This agent retrieves curated travel knowledge and combines it with real-time weather 
and POI data to generate grounded, personalized travel recommendations.
""")

st.divider()

# Input form
col1, col2 = st.columns([2, 1])

with col1:
    city = st.text_input(
        "🌍 Destination City",
        value=os.getenv("TRAVEL_AGENT_DEFAULT_CITY", "Barcelona"),
        help="Enter the city you want to visit"
    )
    prefs = st.text_area(
        "💡 Preferences",
        placeholder="e.g., food, art, nightlife, family-friendly, budget, luxury",
        help="What kind of experience are you looking for?"
    )

with col2:
    days = st.slider("📅 Trip Duration (days)", min_value=1, max_value=14, value=3)
    query = st.text_input(
        "❓ Specific Question (optional)",
        placeholder="e.g., Best time to visit museums?",
        help="Ask a specific question about your trip"
    )

if st.button("🚀 Generate Travel Plan", type="primary", use_container_width=True):
    # Show progress with status
    with st.status("✈️ Liwaa, your AI Travel Agent, is crafting your perfect trip...", expanded=True) as status:
        st.write("🔍 Searching knowledge base for local insights...")
        st.write("🌤️ Checking real-time weather forecast...")
        st.write("📍 Finding the best attractions and hidden gems...")
        
        plan = asyncio.run(agent.plan_trip(
            city=city,
            days=days,
            preferences=prefs or None,
            query=query or None,
        ))
        
        st.write("✍️ Creating your personalized itinerary...")
        status.update(label="✅ Your travel plan is ready!", state="complete", expanded=False)
    
    # Success notification
    st.success(f"🎉 Your {days}-day {city.title()} adventure is ready! Crafted with ❤️ by Liwaa, your AI Travel Agent.")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📍 Destination", city.title())
    
    with col2:
        st.metric("📅 Duration", f"{days} day{'s' if days > 1 else ''}")
    
    with col3:
        confidence_pct = f"{plan.confidence:.0%}"
        st.metric("🎯 Confidence", confidence_pct)
    
    with col4:
        st.metric("📚 Sources", len(plan.sources))
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Itinerary", "🏛️ Attractions", "🌤️ Weather", "📖 Details"])
    
    with tab1:
        structured_days = parse_itinerary(plan.summary)

        if not structured_days:
            # Fallback: Convert markdown bold to HTML and display
            summary_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', plan.summary)
            summary_html = summary_html.replace('\n', '<br>')
            st.markdown(f'<div style="font-size: 15px; line-height: 1.8; color: #333;">{summary_html}</div>', unsafe_allow_html=True)
        else:
            # Render each day as a styled Streamlit container
            for i, day in enumerate(structured_days):
                day_num = i + 1
                day_title = day["title"]
                
                # Day badge + card using columns
                badge_col, card_col = st.columns([0.08, 0.92])
                
                with badge_col:
                    st.markdown(f"""
                    <div style="
                        width: 40px;
                        height: 40px;
                        background: linear-gradient(135deg, #58cc02, #46a302);
                        color: white;
                        border-radius: 50%;
                        text-align: center;
                        line-height: 40px;
                        font-weight: bold;
                        font-size: 18px;
                        box-shadow: 0 3px 6px rgba(88, 204, 2, 0.3);
                        margin-top: 10px;
                    ">{day_num}</div>
                    """, unsafe_allow_html=True)
                
                with card_col:
                    with st.expander(f"🗺️ {day_title}", expanded=True):
                        for slot, desc in day["activities"]:
                            # Icon selection
                            icon = "📍"
                            s_low = slot.lower()
                            if "morning" in s_low: icon = "🌅"
                            elif "lunch" in s_low: icon = "🍽️"
                            elif "afternoon" in s_low: icon = "🏙️"
                            elif "evening" in s_low or "dinner" in s_low: icon = "🌙"
                            elif "pro tip" in s_low or "tip" in s_low: icon = "💡"
                            
                            # Convert **text** to <b>text</b> for HTML rendering
                            desc_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', desc)
                            
                            st.markdown(f"""
                            <div style="
                                display: flex;
                                align-items: flex-start;
                                padding: 14px 0;
                                border-bottom: 1px solid #eee;
                            ">
                                <div style="
                                    font-size: 1.6em;
                                    margin-right: 16px;
                                    min-width: 40px;
                                ">{icon}</div>
                                <div style="flex: 1;">
                                    <div style="
                                        color: #1cb0f6;
                                        font-weight: 700;
                                        font-size: 15px;
                                        margin-bottom: 6px;
                                    ">{slot}</div>
                                    <div style="
                                        color: #333;
                                        font-size: 15px;
                                        line-height: 1.7;
                                    ">{desc_html}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Connector line between days (except last)
                if i < len(structured_days) - 1:
                    st.markdown("""
                    <div style="
                        margin-left: 19px;
                        width: 4px;
                        height: 20px;
                        background: linear-gradient(to bottom, #58cc02, #e5e5e5);
                        border-radius: 2px;
                    "></div>
                    """, unsafe_allow_html=True)

    
    with tab2:
        st.subheader("🏛️ Points of Interest")
        
        if plan.pois:
            # Display POIs in a nice grid
            cols = st.columns(2)
            for idx, poi in enumerate(plan.pois):
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>📍 {poi}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.info(f"ℹ️ Found {len(plan.pois)} attractions near {city.title()}")
        else:
            st.info("No specific POIs found. Check the itinerary for detailed recommendations.")
    
    with tab3:
        st.subheader("🌤️ Weather & Climate")
        
        # Display weather prominently
        st.info(plan.weather)
        
        st.markdown("""
        **Tips for packing:**
        - Check daily forecasts before your trip
        - Pack layers for temperature variations
        - Bring an umbrella if rain is expected
        """)
    
    with tab4:
        st.subheader("📊 Trip Metadata")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Sources:**")
            if plan.sources:
                for source in plan.sources:
                    st.markdown(f"- `{source}`")
            else:
                st.markdown("- General travel knowledge")
        
        with col2:
            st.markdown("**Model Confidence:**")
            
            # Confidence explanation
            if plan.confidence >= 0.8:
                st.success(f"✅ High confidence ({plan.confidence:.0%})")
                st.caption("This plan is well-grounded in our knowledge base.")
            elif plan.confidence >= 0.6:
                st.warning(f"⚠️ Medium confidence ({plan.confidence:.0%})")
                st.caption("Some recommendations are based on general knowledge.")
            else:
                st.error(f"❌ Low confidence ({plan.confidence:.0%})")
                st.caption("Verify details with official sources.")
    
    st.divider()
    
    # Download button for full itinerary
    full_text = f"""
Travel Plan for {city.title()} ({days} days)

Preferences: {prefs if prefs else 'Not specified'}
Confidence: {plan.confidence:.0%}
Sources: {', '.join(plan.sources) if plan.sources else 'General knowledge'}

ITINERARY:
{plan.summary}

WEATHER:
{plan.weather}

POINTS OF INTEREST:
{chr(10).join(f"- {poi}" for poi in plan.pois) if plan.pois else "See itinerary for details."}
"""
    
    st.download_button(
        label="📥 Download Itinerary as Text",
        data=full_text,
        file_name=f"{city}_itinerary_{days}days.txt",
        mime="text/plain",
        use_container_width=True
    )

# Footer
st.divider()
st.caption("Built with LangChain, LangGraph, and ChromaDB | RAG-powered recommendations")
