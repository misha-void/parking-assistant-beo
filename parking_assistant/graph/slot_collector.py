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

Rules:
- If the date is ambiguous, choose the nearest FUTURE date (never a past one).
- If the year is omitted, use the current year (or next year if that date has already passed).
- Use 24-hour time; "9am" -> 09:00:00, "5pm" -> 17:00:00. If no time is given, use 09:00:00.
- Resolve relative terms ("tomorrow", "next Monday") from today's date.

User input: {date_str}

Respond with ONLY the ISO 8601 datetime, nothing else. If it is not a date/time, respond with "INVALID".
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

Pick the single best location ONLY if it genuinely matches the user's description (name, address, or area).
If nothing clearly matches or several fit equally well, do not guess.

Respond with ONLY the number (0, 1, 2, etc.) of the matching location, or "NONE" if no genuine match.
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


# Human-readable labels for the slot the collector is currently asking about
SLOT_LABELS = {
    "name": "first name",
    "surname": "surname",
    "car_number": "car license plate",
    "location": "parking location choice",
    "start_date": "reservation start date/time",
    "end_date": "reservation end date/time",
}


async def infer_location_from_conversation(history: list, current: str, locations: list, llm) -> Optional[int]:
    """
    From the recent conversation (not just the latest message), infer which bookable
    location the user wants. `locations` is [{id, name, address}]. Returns the id or None.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    loc_text = "\n".join(
        f"{i}. {l['name']} - {l.get('address', '')} - "
        f"{str(l['price']) + ' RSD/hour' if l.get('price') else 'price varies'}"
        for i, l in enumerate(locations)
    )
    convo_lines = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in (history or [])[-6:]]
    convo_lines.append(f"User (now): {current}")
    conversation = "\n".join(convo_lines)

    prompt = ChatPromptTemplate.from_template("""
Bookable parking locations:
{locations}

Conversation:
{conversation}

Which ONE of the bookable locations does the user want to book? Consider the whole
conversation, not just the last line. If they ask for the "cheapest"/"most expensive",
choose by the listed RSD/hour price (ignore "price varies" garages). Respond with ONLY
the number, or "NONE" if it is not clear which location they want.
""")
    result = (await (prompt | llm | StrOutputParser()).ainvoke(
        {"locations": loc_text, "conversation": conversation})).strip()

    if result.upper().startswith("NONE"):
        return None
    try:
        idx = int(result)
        if 0 <= idx < len(locations):
            return locations[idx]["id"]
    except ValueError:
        pass
    return None


async def interpret_slot_reply(slot: str, reply: str, history: list, llm,
                               locations: list = None, start_iso: str = None,
                               today: str = None) -> dict:
    """
    Interpret one reservation turn in context. Returns {"kind", "value"}:
      - kind "answer": the user provided the asked slot; "value" is normalized:
          name/surname -> just the name; car_number -> plate (upper, no spaces);
          location -> the EXACT name of the chosen location from `locations`
            (resolving references/affirmations like "the purple one" / "yes" from the
            conversation), or "" if none matches;
          start_date/end_date -> an ISO 8601 datetime (for end_date, resolve "same day"
            and relative terms against the start date).
      - kind "question": the user asked something or didn't answer; "value" is "".
    Classify as "question" only when the user is clearly not answering the asked slot.
    """
    import json
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    loc_text = "\n".join(
        f"- {l['name']} ({str(l['price']) + ' RSD/hour' if l.get('price') else 'price varies'})"
        for l in (locations or [])
    ) or "(none)"
    convo = "\n".join(f"{m.get('role','user').capitalize()}: {m.get('content','')}"
                      for m in (history or [])[-6:]) or "(none)"

    prompt = ChatPromptTemplate.from_template("""
You are helping fill a parking reservation. The user was just asked for their: {slot_label}.
User reply: "{reply}"

Conversation so far:
{conversation}

Bookable locations (for location slot):
{locations}

Today is {today}. Chosen start date/time (for end-date slot): {start_iso}

Decide if the reply ANSWERS the "{slot_label}" or is a QUESTION / not an answer.
Return STRICT JSON: {{"kind": "answer"|"question", "value": "<normalized value or empty>"}}

Normalization when kind is "answer":
- first name / surname: value = just the name (e.g. "Yes, my name is Michke" -> "Michke").
- car license plate: value = the plate, uppercase, no spaces.
- parking location choice: value = the EXACT location name chosen. Resolve references and
  affirmations ("the purple one", "that one", "yes") using the conversation, and
  "cheapest"/"most expensive" by the listed RSD/hour price (ignore "price varies"). "" if unclear.
- start/end date-time: value = ISO 8601 (YYYY-MM-DDTHH:MM:SS); use 24h; assume future; for the
  end date, "same day"/relative phrases are relative to the start date above.
Only use "question" when the reply is clearly a question or refuses/deflects the asked slot.
Respond with ONLY the JSON object.
""")
    raw = (await (prompt | llm | StrOutputParser()).ainvoke({
        "slot_label": SLOT_LABELS.get(slot, slot),
        "reply": reply,
        "conversation": convo,
        "locations": loc_text,
        "today": today or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start_iso": start_iso or "(not set yet)",
    })).strip()

    # Tolerant parse (strip code fences / stray text)
    if "{" in raw:
        raw = raw[raw.index("{"): raw.rindex("}") + 1]
    try:
        data = json.loads(raw)
        kind = "question" if str(data.get("kind", "")).lower().startswith("q") else "answer"
        return {"kind": kind, "value": str(data.get("value") or "").strip()}
    except (ValueError, KeyError):
        return {"kind": "answer", "value": reply.strip()}  # safe fallback: treat as answer


async def is_question_not_answer(reply: str, slot: str, llm) -> bool:
    """
    During slot collection, decide whether the user's reply is answering the asked
    slot or instead asking a question / making a different request.

    Returns True if the reply is a question/other request (should be answered, not stored).
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template("""
A user is booking a parking spot and was just asked to provide their {slot_label}.

Their reply: "{reply}"

Is the reply actually providing that {slot_label}, or is it a question / a different request
(e.g. asking about addresses, prices, which option to pick)?

Respond with ONLY one word: ANSWER or QUESTION.
""")
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({"slot_label": SLOT_LABELS.get(slot, slot), "reply": reply})
    return "QUESTION" in result.strip().upper()
