"""Slot-filling conversation logic for parking reservations."""

from typing import Dict, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
import re


class ReservationSlots(BaseModel):
    """Data structure for reservation slot values."""
    name: Optional[str] = None
    surname: Optional[str] = None
    car_number: Optional[str] = None
    location_preference: Optional[str] = None  # User's description
    location_id: Optional[int] = None  # Matched location ID
    start_date: Optional[str] = None  # ISO format or natural language
    end_date: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if all required slots are filled."""
        return all([
            self.name,
            self.surname,
            self.car_number,
            self.location_id,
            self.start_date,
            self.end_date
        ])
    
    def missing_slots(self) -> List[str]:
        """Return list of missing slot names."""
        missing = []
        if not self.name:
            missing.append("name")
        if not self.surname:
            missing.append("surname")
        if not self.car_number:
            missing.append("car_number")
        if not self.location_id:
            missing.append("location")
        if not self.start_date:
            missing.append("start_date")
        if not self.end_date:
            missing.append("end_date")
        return missing


class SlotCollector:
    """Manages slot-filling conversation for reservations."""
    
    # Slot collection prompts
    SLOT_PROMPTS = {
        "name": "What's your first name?",
        "surname": "What's your surname (last name)?",
        "car_number": "What's your car license plate number?",
        "location": "Which parking location would you prefer? (You can describe it or I can show you available options)",
        "start_date": "When would you like to start your reservation? (e.g., 'tomorrow at 9am', 'December 25 at 10:00')",
        "end_date": "When would you like to end your reservation? (e.g., 'same day at 5pm', 'December 25 at 18:00')"
    }
    
    def __init__(self):
        # Require at least one digit (real plates are never pure letters) plus
        # allow letters/hyphens, length 3-10. Using a lookahead so "invalid"
        # (letters only) is correctly rejected while "BG123AB" / "ABC-123" pass.
        self.car_number_pattern = re.compile(r'^(?=.*[0-9])[A-Z0-9\-]{3,10}$', re.IGNORECASE)
    
    def extract_slots_from_message(self, message: str, current_slots: ReservationSlots) -> ReservationSlots:
        """
        Extract slot values from user message.
        Simple keyword/pattern matching for now.
        """
        message_lower = message.lower()
        
        # Extract car number (if looks like plate)
        if not current_slots.car_number:
            # Look for patterns like BG123AB, 123-ABC, etc.
            tokens = message.split()
            for token in tokens:
                if self.car_number_pattern.match(token):
                    current_slots.car_number = token.upper()
                    break
        
        # For names, if we just asked for name and got a short response, assume it's the name
        # (This is simplified - in production would use NER or LLM extraction)
        
        return current_slots
    
    def validate_car_number(self, car_number: str) -> bool:
        """Validate car number format."""
        return self.car_number_pattern.match(car_number) is not None
    
    def get_next_prompt(self, slots: ReservationSlots) -> str:
        """Get the next question to ask based on missing slots."""
        missing = slots.missing_slots()
        
        if not missing:
            return None  # All slots filled
        
        # Return prompt for first missing slot
        next_slot = missing[0]
        return self.SLOT_PROMPTS[next_slot]
    
    def format_confirmation(self, slots: ReservationSlots, location_name: str) -> str:
        """Format reservation details for user confirmation."""
        return f"""
📋 **Reservation Summary**

👤 Name: {slots.name} {slots.surname}
🚗 Car: {slots.car_number}
📍 Location: {location_name}
📅 Start: {slots.start_date}
📅 End: {slots.end_date}

Is this information correct? (Reply 'yes' to confirm or 'no' to cancel)
""".strip()


async def parse_datetime_with_llm(date_str: str, llm) -> Optional[datetime]:
    """
    Use LLM to parse natural language date/time into datetime object.
    
    Args:
        date_str: User's date input (e.g., "tomorrow at 9am", "Dec 25 at 10:00")
        llm: Language model instance
    
    Returns:
        Parsed datetime or None if unparseable
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    prompt = ChatPromptTemplate.from_template("""
Parse this date/time expression into ISO 8601 format (YYYY-MM-DDTHH:MM:SS).
Today is {today}.

User input: {date_str}

Respond with ONLY the ISO datetime, nothing else. If unparseable, respond with "INVALID".
""")
    
    chain = prompt | llm | StrOutputParser()
    
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = await chain.ainvoke({"date_str": date_str, "today": today})
    
    result = result.strip()
    
    if result == "INVALID" or not result:
        return None
    
    try:
        # Try parsing ISO format
        return datetime.fromisoformat(result.replace('Z', '+00:00'))
    except ValueError:
        return None


async def match_location_with_llm(preference: str, available_locations: List[Dict], llm) -> Optional[int]:
    """
    Match user's location preference to available locations using LLM.
    
    Args:
        preference: User's description (e.g., "near city center", "UŠĆE mall")
        available_locations: List of dicts with name, address, description
        llm: Language model instance
    
    Returns:
        Location ID (index in list) or None if no good match
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    # Format locations for prompt
    locations_text = "\n".join([
        f"{i}. {loc['name']} - {loc['address']}"
        for i, loc in enumerate(available_locations)
    ])
    
    prompt = ChatPromptTemplate.from_template("""
Match the user's parking preference to one of these locations:

{locations}

User preference: {preference}

Respond with ONLY the number (0, 1, 2, etc.) of the best matching location.
If no good match, respond with "NONE".
""")
    
    chain = prompt | llm | StrOutputParser()
    
    result = await chain.ainvoke({
        "locations": locations_text,
        "preference": preference
    })
    
    result = result.strip()
    
    if result == "NONE":
        return None
    
    try:
        idx = int(result)
        if 0 <= idx < len(available_locations):
            return available_locations[idx]['id']
    except ValueError:
        pass
    
    return None
